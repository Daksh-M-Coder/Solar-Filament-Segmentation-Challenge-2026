"""
Task 2 – Contamination Screen (Frangi vesselness filter)
=========================================================
HN mask PNGs are all-zeros by design (they were mined as FPs with empty masks).
So we score filament-likeness using the FULL tile Frangi response (top-10% mean),
and calibrate the p80 threshold against the same statistic computed on 100
randomly-sampled GT filament TILE crops from the training set.

This gives a proper apples-to-apples comparison:
  GT crop  → Frangi(full crop)  top-10% mean
  HN tile  → Frangi(full tile)  top-10% mean

Outputs:
  - data/hard_negatives_v1/possible_contamination/  (flagged tile+mask copies)
  - reports/hn_contamination_manifest.json
  - reports/hn_contamination_montage.png  (top-30 by Frangi score)
  - reports/hn_clean_montage.png          (30 random clean tiles)
"""

import json, os, sys, shutil, random
import numpy as np
import cv2
from skimage.filters import frangi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST_PATH     = "data/hard_negatives_v1/hard_negatives_manifest.json"
HN_DIR            = "data/hard_negatives_v1"
CONTAMINATION_DIR = "data/hard_negatives_v1/possible_contamination"
COCO_JSON         = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES_DIR  = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
OUT_MANIFEST      = "reports/hn_contamination_manifest.json"
OUT_MONTAGE_F     = "reports/hn_contamination_montage.png"
OUT_MONTAGE_C     = "reports/hn_clean_montage.png"

FRANGI_SIGMAS     = (1, 2, 4, 6)
FRANGI_ALPHA      = 0.5
FRANGI_BETA       = 0.5
# solar filaments appear dark on a bright background
FRANGI_BLACK_RIDGES = True
CROP_SIZE         = 512   # GT crops resized to this before Frangi
TOP_PCT           = 0.10  # use top 10% of Frangi response as the score
THUMB_SIZE        = 128
GRID_COLS         = 5


def frangi_tile_score(img_gray: np.ndarray) -> float:
    """
    Top-10% mean Frangi response across the full image/crop.
    Works on both GT bbox crops and HN full tiles.
    """
    img_f = img_gray.astype(np.float64) / 255.0
    resp  = frangi(img_f, sigmas=FRANGI_SIGMAS, alpha=FRANGI_ALPHA,
                   beta=FRANGI_BETA, black_ridges=FRANGI_BLACK_RIDGES)
    flat  = resp.ravel()
    if len(flat) == 0:
        return 0.0
    thresh = np.percentile(flat, (1.0 - TOP_PCT) * 100)
    top_vals = flat[flat >= thresh]
    return float(np.mean(top_vals)) if len(top_vals) > 0 else 0.0


def build_gt_distribution(coco_json, images_dir, max_samples=100):
    """
    For each sampled GT annotation, extract the bbox crop from the training image
    (resized to CROP_SIZE x CROP_SIZE), compute frangi_tile_score, return array.
    """
    print(f"[GT dist] Scoring {max_samples} GT bbox crops …", flush=True)
    with open(coco_json) as f:
        coco = json.load(f)
    from pycocotools import mask as mask_utils
    img_map = {img["id"]: img for img in coco["images"]}
    anns    = coco["annotations"]
    random.seed(42)
    sampled = random.sample(anns, min(max_samples, len(anns)))
    scores  = []

    for i, ann in enumerate(sampled):
        info = img_map.get(ann["image_id"])
        if info is None:
            continue
        img = cv2.imread(os.path.join(images_dir, info["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = info["height"], info["width"]
        seg  = ann.get("segmentation", [])
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, h, w)
            m    = mask_utils.decode(mask_utils.merge(rles))
        elif isinstance(seg, dict):
            m = mask_utils.decode(seg)
        else:
            continue
        if np.sum(m) < 50:
            continue
        # Crop to bbox + 32px pad, resize to CROP_SIZE
        rs, cs = np.where(m)
        pad    = 32
        r0, r1 = max(0, rs.min()-pad), min(h, rs.max()+pad+1)
        c0, c1 = max(0, cs.min()-pad), min(w, cs.max()+pad+1)
        crop   = img[r0:r1, c0:c1]
        if crop.size == 0:
            continue
        crop_rs = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        sc = frangi_tile_score(crop_rs)
        scores.append(sc)
        if (i + 1) % 10 == 0:
            print(f"  GT [{i+1}/{len(sampled)}] score={sc:.5f}", flush=True)

    arr = np.array(scores, dtype=np.float64)
    if len(arr) > 0:
        print(f"[GT dist] Done — {len(arr)} crops. "
              f"min={arr.min():.5f} median={np.median(arr):.5f} "
              f"p80={np.percentile(arr,80):.5f} max={arr.max():.5f}", flush=True)
    return arr


def make_montage(paths, out_path):
    n = len(paths)
    if n == 0:
        return
    rows   = (n + GRID_COLS - 1) // GRID_COLS
    canvas = np.zeros((rows * THUMB_SIZE, GRID_COLS * THUMB_SIZE, 3), np.uint8)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (THUMB_SIZE, THUMB_SIZE))
        r, c = divmod(i, GRID_COLS)
        canvas[r*THUMB_SIZE:(r+1)*THUMB_SIZE, c*THUMB_SIZE:(c+1)*THUMB_SIZE] = img
    cv2.imwrite(out_path, canvas)
    print(f"  Montage → {out_path}", flush=True)


def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs(CONTAMINATION_DIR, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        mn = json.load(f)
    entries = mn["manifest"]
    print(f"HN pool: {len(entries)} tiles", flush=True)
    print(f"Scoring method: top-{int(TOP_PCT*100)}% mean Frangi on full tile/crop", flush=True)
    print(f"Frangi params: sigmas={FRANGI_SIGMAS}, black_ridges={FRANGI_BLACK_RIDGES}\n", flush=True)

    # Sanity: score first HN tile
    first   = entries[0]
    test_ti = cv2.imread(os.path.join(HN_DIR, first["tile_file"]), cv2.IMREAD_GRAYSCALE)
    if test_ti is not None:
        san = frangi_tile_score(test_ti)
        print(f"[SANITY] First HN tile top-10% Frangi: {san:.6f}", flush=True)

    # GT distribution
    gt_sc = build_gt_distribution(COCO_JSON, TRAIN_IMAGES_DIR, max_samples=100)
    if len(gt_sc) == 0:
        print("ERROR: empty GT distribution — aborting", flush=True)
        return
    p80 = float(np.percentile(gt_sc, 80))
    print(f"\nGT p80 threshold: {p80:.5f}", flush=True)

    # Score all 500 HN tiles
    results, flagged_paths, clean_paths = [], [], []
    flagged = 0

    print(f"\nScoring {len(entries)} HN tiles …", flush=True)
    for i, e in enumerate(entries):
        tf = os.path.join(HN_DIR, e["tile_file"])
        mf = os.path.join(HN_DIR, e["mask_file"])
        ti = cv2.imread(tf, cv2.IMREAD_GRAYSCALE)
        if ti is None:
            print(f"  WARN: missing {e['tile_file']}", flush=True)
            continue
        sc = frangi_tile_score(ti)
        fl = sc > p80
        results.append({
            "rank": e["rank"], "tile_file": e["tile_file"], "mask_file": e["mask_file"],
            "source_image": e["source_image"], "area": e["area"],
            "confidence": e["confidence"], "frangi_score": sc,
            "p80_threshold": p80, "flagged": fl,
        })
        if fl:
            flagged += 1
            shutil.copy(tf, os.path.join(CONTAMINATION_DIR, e["tile_file"]))
            if os.path.exists(mf):
                shutil.copy(mf, os.path.join(CONTAMINATION_DIR, e["mask_file"]))
            flagged_paths.append(tf)
        else:
            clean_paths.append(tf)

        if (i + 1) % 50 == 0 or (i + 1) == len(entries):
            hn_so_far = [r["frangi_score"] for r in results]
            print(f"  HN [{i+1}/{len(entries)}] flagged={flagged} "
                  f"| running max={max(hn_so_far):.5f} min={min(hn_so_far):.5f}", flush=True)

    total     = len(results)
    unflagged = total - flagged
    hn_scores = [r["frangi_score"] for r in results]

    # Save manifest
    with open(OUT_MANIFEST, "w") as f:
        json.dump({
            "total_tiles":       total,
            "flagged_tiles":     flagged,
            "unflagged_tiles":   unflagged,
            "p80_threshold":     p80,
            "gt_scores_used":    len(gt_sc),
            "gt_frangi_stats":   {"min": float(gt_sc.min()), "median": float(np.median(gt_sc)),
                                  "p80": p80, "max": float(gt_sc.max())},
            "hn_frangi_stats":   {"min":    float(np.min(hn_scores)),
                                  "median": float(np.median(hn_scores)),
                                  "p80":    float(np.percentile(hn_scores, 80)),
                                  "max":    float(np.max(hn_scores))},
            "frangi_params":     {"sigmas": list(FRANGI_SIGMAS), "alpha": FRANGI_ALPHA,
                                  "beta": FRANGI_BETA, "black_ridges": FRANGI_BLACK_RIDGES,
                                  "score_method": f"top-{int(TOP_PCT*100)}%-mean"},
            "results": sorted(results, key=lambda x: x["frangi_score"], reverse=True),
        }, f, indent=2)
    print(f"Manifest saved → {OUT_MANIFEST}", flush=True)

    # Montages
    top30 = [os.path.join(HN_DIR, r["tile_file"])
             for r in sorted(results, key=lambda x: x["frangi_score"], reverse=True)
             if r["flagged"]][:30]
    make_montage(top30, OUT_MONTAGE_F)

    random.seed(99)
    make_montage(random.sample(clean_paths, min(30, len(clean_paths))), OUT_MONTAGE_C)

    # Summary
    print("\n" + "="*58)
    print("CONTAMINATION SCREEN SUMMARY")
    print("="*58)
    print(f"  Scoring method    : top-{int(TOP_PCT*100)}% Frangi on full tile")
    print(f"  GT p80 threshold  : {p80:.5f}")
    print(f"  GT Frangi range   : {gt_sc.min():.5f} – {gt_sc.max():.5f}")
    print(f"  HN Frangi range   : {np.min(hn_scores):.5f} – {np.max(hn_scores):.5f}")
    print(f"  HN Frangi median  : {np.median(hn_scores):.5f}")
    print(f"  Total tiles       : {total}")
    print(f"  Flagged (≥ p80)   : {flagged}  ({100*flagged/max(total,1):.1f}%)")
    print(f"  Unflagged         : {unflagged}  ({100*unflagged/max(total,1):.1f}%)")
    print(f"  Contamination dir : {CONTAMINATION_DIR}")
    print(f"  Manifest          : {OUT_MANIFEST}")
    print(f"  Montage (flagged) : {OUT_MONTAGE_F}")
    print(f"  Montage (clean)   : {OUT_MONTAGE_C}")
    print("="*58)


if __name__ == "__main__":
    main()
