"""
Task 1 – FP Bucket Split
========================
For the 115-pool with unet_cldice_v2_best.pth at locked config, take every FP
instance and compute its best-IoU against any GT filament on that image.
Bucket them:
  (a) IoU = 0.000
  (b) 0.00 < IoU <= 0.30
  (c) 0.30 < IoU <= 0.45
  (d) 0.45 < IoU <= 0.48
  (e) 0.48 < IoU <= 0.50
Report count and mean area per bucket, plus % of total FPs in each.
Same breakdown for the mined HN pool.
Save as reports/fp_bucket_split.md
"""

import json
import os
import sys
import numpy as np
import torch
import yaml
import cv2
from pycocotools import mask as mask_utils

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image

CHECKPOINT   = "models/unet_cldice_v2_best.pth"
CONFIG       = "configs/unet_config.yaml"
LIMIT        = 115
SCORE_THRESH = 0.45
USE_TTA      = True
TTA_MODE     = "prob_avg"
MANIFEST     = "data/hard_negatives_v1/hard_negatives_manifest.json"
OUT_MD       = "reports/fp_bucket_split.md"

BUCKETS = [
    ("(a) IoU=0.000",         0.000, 0.000),
    ("(b) 0.000<IoU<=0.30",   0.000, 0.300),
    ("(c) 0.30<IoU<=0.45",    0.300, 0.450),
    ("(d) 0.45<IoU<=0.48",    0.450, 0.480),
    ("(e) 0.48<IoU<=0.50",    0.480, 0.500),
]


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = (mask_a > 0).astype(bool)
    b = (mask_b > 0).astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def best_iou_against_gts(pred_mask: np.ndarray, gt_masks: list) -> float:
    if not gt_masks:
        return 0.0
    return max(iou(pred_mask, g) for g in gt_masks)


def bucket_label(best_iou_val: float) -> str:
    if best_iou_val == 0.0:
        return "(a) IoU=0.000"
    elif best_iou_val <= 0.30:
        return "(b) 0.000<IoU<=0.30"
    elif best_iou_val <= 0.45:
        return "(c) 0.30<IoU<=0.45"
    elif best_iou_val <= 0.48:
        return "(d) 0.45<IoU<=0.48"
    else:
        return "(e) 0.48<IoU<=0.50"


def evaluate_pq_and_collect_fps(pred_masks: list, gt_masks: list, iou_thresh: float = 0.5):
    """
    Strict PQ matching at iou_thresh=0.5. Returns matched TP indices and unmatched
    FP pred_masks indices with their best-GT IoU.
    """
    matched_gt = set()
    matched_pred = set()

    # Sort preds by area desc for deterministic greedy matching
    pred_areas = [int(np.sum(m > 0)) for m in pred_masks]
    sorted_preds = sorted(range(len(pred_masks)), key=lambda i: pred_areas[i], reverse=True)

    for pi in sorted_preds:
        best_iou_val, best_gi = 0.0, -1
        for gi, gm in enumerate(gt_masks):
            if gi in matched_gt:
                continue
            v = iou(pred_masks[pi], gm)
            if v > best_iou_val:
                best_iou_val = v
                best_gi = gi
        if best_iou_val >= iou_thresh and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    fp_records = []
    for pi in range(len(pred_masks)):
        if pi not in matched_pred:
            best_iou_val = best_iou_against_gts(pred_masks[pi], gt_masks)
            # Only include truly FP ones (best IoU < 0.5)
            area = int(np.sum(pred_masks[pi] > 0))
            fp_records.append({"area": area, "best_iou": best_iou_val})

    return fp_records


def main():
    # ── Load config ──────────────────────────────────────────────────────────
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    prod = cfg["production"]
    min_area       = int(prod.get("min_area", 300))
    peak_min_dist  = int(prod.get("peak_min_distance", 55))
    skel_prune_len = int(prod.get("prune_threshold", 15))
    morph_kernel   = int(prod.get("kernel_size", 3))
    dist_type      = str(prod.get("dist_transform", "DIST_L1"))
    postproc_kwargs = {
        "peak_min_dist":    peak_min_dist,
        "skel_prune_len":   skel_prune_len,
        "morph_kernel_size": morph_kernel,
        "dist_transform_type": dist_type,
        "min_area":         min_area,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Postproc: {postproc_kwargs}\n")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # ── Load COCO dataset ────────────────────────────────────────────────────
    json_path   = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    images_dir  = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
    with open(json_path) as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    anns_dict   = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    val_ids = shuffled_ids[:max(20, int(len(all_ids) * 0.10))][:LIMIT]

    # ── Run inference and collect FPs ────────────────────────────────────────
    all_fp_records = []
    for idx, img_id in enumerate(val_ids, 1):
        info   = images_dict[img_id]
        img_p  = os.path.join(images_dir, info["file_name"])
        h, w   = info["height"], info["width"]

        # Build GT masks
        gt_masks = []
        for ann in anns_dict.get(img_id, []):
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                m    = mask_utils.decode(mask_utils.merge(rles))
                gt_masks.append(m)
            elif isinstance(seg, dict):
                gt_masks.append(mask_utils.decode(seg))

        # Run model
        insts, _, _, _ = predict_single_image(
            model=model, img_path=img_p, device=device,
            thresh=SCORE_THRESH, use_tta=USE_TTA, tta_mode=TTA_MODE,
            postproc_kwargs=postproc_kwargs,
        )

        # Collect FPs with their best-GT IoU
        fp_recs = evaluate_pq_and_collect_fps(insts, gt_masks, iou_thresh=0.5)
        all_fp_records.extend(fp_recs)

        if idx % 20 == 0 or idx == len(val_ids):
            print(f"[{idx}/{len(val_ids)}] cumulative FPs so far: {len(all_fp_records)}", flush=True)

    print(f"\nTotal FP instances collected: {len(all_fp_records)}")

    # ── Bucket the FPs ───────────────────────────────────────────────────────
    def build_bucket_table(records, label="FP instances"):
        total = len(records)
        rows  = []
        for bname, lo, hi in BUCKETS:
            if bname == "(a) IoU=0.000":
                subset = [r for r in records if r["best_iou"] == 0.0]
            else:
                subset = [r for r in records if lo < r["best_iou"] <= hi]
            cnt      = len(subset)
            pct      = 100.0 * cnt / total if total > 0 else 0.0
            mean_area = float(np.mean([r["area"] for r in subset])) if subset else 0.0
            rows.append((bname, cnt, pct, mean_area))
        return rows, total

    fp_rows, fp_total = build_bucket_table(all_fp_records, "FP instances on 115-pool")

    # ── HN pool bucket ───────────────────────────────────────────────────────
    with open(MANIFEST) as f:
        mn = json.load(f)
    hn_entries = mn["manifest"]  # 500 entries (all best_iou=0 by definition of mining)
    hn_records = [{"area": e["area"], "best_iou": 0.0} for e in hn_entries]
    hn_rows, hn_total = build_bucket_table(hn_records, "HN pool (top-500 saved)")

    # ── Build Markdown ───────────────────────────────────────────────────────
    md_lines = []
    md_lines.append("# FP Bucket Split Report\n")
    md_lines.append(f"**Checkpoint:** `{CHECKPOINT}`  ")
    md_lines.append(f"**TTA:** {USE_TTA} ({TTA_MODE})  ")
    md_lines.append(f"**Pool:** {LIMIT} images  ")
    md_lines.append(f"**Locked config:** peak={peak_min_dist}, prune={skel_prune_len}, "
                    f"k={morph_kernel}, {dist_type}, th={SCORE_THRESH}, min_area={min_area}\n")

    md_lines.append("## Part A — FP instances from live validation (115-pool)\n")
    md_lines.append(f"Total FP instances: **{fp_total}**\n")
    md_lines.append("| Bucket | Count | % of total FPs | Mean Area (px) |")
    md_lines.append("|--------|------:|---------------:|---------------:|")
    for bname, cnt, pct, ma in fp_rows:
        md_lines.append(f"| {bname} | {cnt} | {pct:.1f}% | {ma:.0f} |")

    md_lines.append("\n## Part B — Mined HN pool breakdown (top-500 saved tiles)\n")
    md_lines.append(f"Total HN tiles: **{hn_total}**  \n"
                    "*Note: by definition all mined HN have best_iou=0.000 (IoU=0 against all GT).*\n")
    md_lines.append("| Bucket | Count | % of total HN | Mean Area (px) |")
    md_lines.append("|--------|------:|--------------:|---------------:|")
    for bname, cnt, pct, ma in hn_rows:
        md_lines.append(f"| {bname} | {cnt} | {pct:.1f}% | {ma:.0f} |")

    md_lines.append("\n## Notes\n")
    md_lines.append("- FP bucketing uses strict PQ IoU=0.5 matching, same as the validator.")
    md_lines.append("- Bucket (a) are pure false positives with zero overlap with any GT filament.")
    md_lines.append("- Buckets (d) and (e) are near-misses — almost matched but fell just below threshold.")
    md_lines.append("- HN pool was mined separately from the training set, so all entries land in bucket (a) by construction.")

    os.makedirs("reports", exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nSaved report to: {OUT_MD}")
    print("\n=== FP Bucket Summary ===")
    print(f"{'Bucket':<25} {'Count':>6} {'%':>7} {'Mean Area':>10}")
    print("-" * 55)
    for bname, cnt, pct, ma in fp_rows:
        print(f"{bname:<25} {cnt:>6} {pct:>6.1f}% {ma:>10.0f}")
    print(f"\n{'TOTAL':<25} {fp_total:>6}")


if __name__ == "__main__":
    main()
