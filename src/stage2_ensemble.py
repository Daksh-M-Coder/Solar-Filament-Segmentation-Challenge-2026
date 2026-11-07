"""
Stage 2: Strict-Gated U-Net + Mask R-CNN Ensemble

Rules:
  - U-Net detections: always kept (primary predictor)
  - Mask R-CNN detection is added ONLY if:
      1. IoU with every U-Net mask < 0.10 (truly complementary)
      2. Mask R-CNN confidence score > 0.85 (high confidence only)
  - Final overlap resolution applied after merge

This script is self-contained (doesn't rely on predict_tiled_maskrcnn from
audit_maskrcnn_115.py) and returns scores alongside masks.

Lazy imports (inside function bodies) avoid Win32 deadlock.
"""
import os
import sys
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────────────────────────────
# IoU / PQ helpers
# ──────────────────────────────────────────────────────────────────────

def compute_mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(m1, m2).sum())


def strict_pq_eval(preds, gts, iou_thresh=0.50):
    n_p, n_g = len(preds), len(gts)
    if n_p == 0 and n_g == 0:
        return {"pq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if n_p == 0:
        return {"pq": 0.0, "tp": 0, "fp": 0, "fn": n_g}
    if n_g == 0:
        return {"pq": 0.0, "tp": 0, "fp": n_p, "fn": 0}
    iou_mat = np.zeros((n_p, n_g), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_mat[i, j] = compute_mask_iou(p, g)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    tp_ious, matched_gts, matched_preds = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_thresh:
            tp_ious.append(iou_mat[r, c])
            matched_gts.add(c)
            matched_preds.add(r)
    tp = len(matched_gts)
    fp = n_p - tp
    fn = n_g - tp
    sq = float(np.mean(tp_ious)) if tp > 0 else 0.0
    rq = float(tp) / float(tp + 0.5 * fp + 0.5 * fn)
    return {"pq": sq * rq, "tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────────────────────────────
# Mask R-CNN tiled inference WITH scores
# ──────────────────────────────────────────────────────────────────────

def _get_tile_coords(H, W, tile_size=512, overlap=64):
    step = tile_size - overlap
    ys = list(range(0, H - tile_size + 1, step))
    if not ys or ys[-1] != H - tile_size:
        ys.append(H - tile_size)
    xs = list(range(0, W - tile_size + 1, step))
    if not xs or xs[-1] != W - tile_size:
        xs.append(W - tile_size)
    return [(y, x, y + tile_size, x + tile_size) for y in ys for x in xs]


def predict_mrcnn_with_scores(model, image_tensor, device,
                              tile_size=512, overlap=64,
                              score_thresh=0.50, mask_thresh=0.50,
                              nms_thresh=0.50):
    """Run tiled Mask R-CNN inference; return (masks, scores)."""
    from torchvision.ops import nms

    _, H, W = image_tensor.shape
    tile_coords = _get_tile_coords(H, W, tile_size, overlap)

    global_boxes, global_scores, global_masks = [], [], []
    for ymin, xmin, ymax, xmax in tile_coords:
        tile_t = image_tensor[:, ymin:ymax, xmin:xmax]
        with torch.no_grad():
            out = model([tile_t.to(device)])[0]
        scores = out["scores"].cpu().numpy()
        masks = out["masks"].squeeze(1).cpu().numpy()
        labels = out["labels"].cpu().numpy()
        boxes = out["boxes"].cpu().numpy()

        keep = [i for i, (s, l) in enumerate(zip(scores, labels))
                if s >= score_thresh and l == 1]
        for idx in keep:
            sc = float(scores[idx])
            b = boxes[idx]
            m = (masks[idx] > mask_thresh).astype(np.uint8)
            g_box = [b[0]+xmin, b[1]+ymin, b[2]+xmin, b[3]+ymin]
            g_mask = np.zeros((H, W), dtype=np.uint8)
            g_mask[ymin:ymax, xmin:xmax] = m
            global_boxes.append(g_box)
            global_scores.append(sc)
            global_masks.append(g_mask)

    if len(global_boxes) == 0:
        return [], []

    t_boxes = torch.tensor(np.array(global_boxes), dtype=torch.float32)
    t_scores = torch.tensor(np.array(global_scores), dtype=torch.float32)
    keep_idx = nms(t_boxes, t_scores, iou_threshold=nms_thresh).numpy()

    final_masks = [global_masks[k] for k in keep_idx]
    final_scores = [global_scores[k] for k in keep_idx]
    return final_masks, final_scores


# ──────────────────────────────────────────────────────────────────────
# Gating & overlap resolution
# ──────────────────────────────────────────────────────────────────────

def strict_gate_mrcnn(unet_insts, mrcnn_insts, mrcnn_scores,
                      max_iou_with_unet=0.10, min_score=0.85):
    gated, gated_scores = [], []
    for inst, score in zip(mrcnn_insts, mrcnn_scores):
        if score < min_score:
            continue
        max_iou = max(
            (compute_mask_iou(inst, u) for u in unet_insts),
            default=0.0
        )
        if max_iou < max_iou_with_unet:
            gated.append(inst)
            gated_scores.append(score)
    return gated, gated_scores


def resolve_overlaps_greedy(instances, min_area=300):
    if len(instances) <= 1:
        return [i for i in instances if np.sum(i > 0) >= min_area]
    areas = [int(np.sum(inst > 0)) for inst in instances]
    order = np.argsort(areas)[::-1]
    H, W = instances[0].shape
    claimed = np.zeros((H, W), dtype=bool)
    result = []
    for idx in order:
        mask = (instances[idx] > 0)
        unclaimed = mask & (~claimed)
        if int(np.sum(unclaimed)) >= min_area:
            result.append(unclaimed.astype(np.uint8))
            claimed |= unclaimed
    return result


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def run_stage2_ensemble(
    json_path="data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir="data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    cache_tta_dir="data/cache_tta_val115",
    cache_mrcnn_dir="data/cache_mrcnn_val115",
    mrcnn_weights="models/best_model.pth",
    unet_thresh=0.45,
    unet_postproc_kwargs=None,
    mrcnn_score_thresh=0.85,
    mrcnn_iou_gate=0.10,
    min_area=300,
    output_report="reports/stage2_ensemble.json",
):
    # Lazy import to avoid deadlock
    from src.unet_postprocess import postprocess_binary_mask
    from src.audit_maskrcnn_115 import build_mask_rcnn

    if unet_postproc_kwargs is None:
        unet_postproc_kwargs = {
            "peak_min_dist": 55,
            "skel_prune_len": 15,
            "morph_kernel_size": 3,
            "dist_transform_type": "DIST_L1",
            "min_area": min_area,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"U-Net thresh: {unet_thresh}", flush=True)
    print(f"Mask R-CNN gating: score > {mrcnn_score_thresh}, IoU with U-Net < {mrcnn_iou_gate}", flush=True)

    # Load 115-pool IDs
    with open(json_path, "r") as f:
        coco = json.load(f)
    images_dict = {img["id"]: img for img in coco["images"]}
    anns_dict = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    val_ids = shuffled_ids[:max(20, int(len(all_ids) * 0.10))]
    print(f"Validation pool: {len(val_ids)} images", flush=True)

    # Mask R-CNN model (lazy load on first cache miss)
    mrcnn_model = None
    os.makedirs(cache_mrcnn_dir, exist_ok=True)

    unet_pqs, ens_pqs = [], []
    unet_tp, unet_fp, unet_fn = 0, 0, 0
    ens_tp, ens_fp, ens_fn = 0, 0, 0
    gated_added_total = 0

    for img_id in tqdm(val_ids, desc="Stage 2 Ensemble"):
        info = images_dict[img_id]
        h, w = info["height"], info["width"]
        img_p = os.path.join(images_dir, info["file_name"])

        # Ground truth
        gt_instances = []
        for ann in anns_dict.get(img_id, []):
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(mask_utils.merge(rles))
                gt_instances.append(m)
            elif isinstance(seg, dict):
                m = mask_utils.decode(seg)
                gt_instances.append(m)

        # U-Net predictions (cached TTA prob maps)
        tta_cache_p = os.path.join(cache_tta_dir, f"{img_id}.npz")
        prob_map = np.load(tta_cache_p)["prob"].astype(np.float32)
        unet_insts = postprocess_binary_mask(prob_map, thresh=unet_thresh, **unet_postproc_kwargs)

        # Mask R-CNN predictions
        masks_cache = os.path.join(cache_mrcnn_dir, f"{img_id}.npy")
        scores_cache = os.path.join(cache_mrcnn_dir, f"{img_id}_scores.npy")

        if os.path.exists(masks_cache) and os.path.exists(scores_cache):
            mrcnn_insts = list(np.load(masks_cache, allow_pickle=True))
            mrcnn_scores = list(np.load(scores_cache))
        else:
            # Need to run inference — load model if not loaded
            if mrcnn_model is None:
                alt = "archive/Mask R-CNN/models/best_model.pth"
                weights = mrcnn_weights if os.path.exists(mrcnn_weights) else alt
                print(f"Loading Mask R-CNN from {weights}...", flush=True)
                mrcnn_model = build_mask_rcnn(num_classes=2, box_nms_thresh=0.5)
                ckpt = torch.load(weights, map_location=device)
                state = ckpt.get("model_state_dict", ckpt)
                mrcnn_model.load_state_dict(state)
                mrcnn_model.to(device)
                mrcnn_model.eval()

            img_bgr = cv2.imread(img_p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            mrcnn_insts, mrcnn_scores = predict_mrcnn_with_scores(
                mrcnn_model, img_tensor, device,
                score_thresh=0.50, mask_thresh=0.50, nms_thresh=0.50
            )
            # Cache with scores
            bool_masks = np.array([(m > 0).astype(np.uint8) for m in mrcnn_insts], dtype=object)
            np.save(masks_cache, bool_masks, allow_pickle=True)
            np.save(scores_cache, np.array(mrcnn_scores, dtype=np.float32))

        mrcnn_scores = [float(s) for s in mrcnn_scores]

        # Strict gating
        gated_mrcnn, _ = strict_gate_mrcnn(
            unet_insts, mrcnn_insts, mrcnn_scores,
            max_iou_with_unet=mrcnn_iou_gate,
            min_score=mrcnn_score_thresh
        )
        gated_added_total += len(gated_mrcnn)

        # Merge & resolve overlaps
        merged = list(unet_insts) + gated_mrcnn
        ensemble_insts = resolve_overlaps_greedy(merged, min_area=min_area)

        # Evaluate
        u_res = strict_pq_eval(unet_insts, gt_instances)
        e_res = strict_pq_eval(ensemble_insts, gt_instances)

        unet_pqs.append(u_res["pq"])
        ens_pqs.append(e_res["pq"])
        unet_tp += u_res["tp"]; unet_fp += u_res["fp"]; unet_fn += u_res["fn"]
        ens_tp += e_res["tp"]; ens_fp += e_res["fp"]; ens_fn += e_res["fn"]

    mean_unet_pq = float(np.mean(unet_pqs))
    mean_ens_pq = float(np.mean(ens_pqs))
    delta = mean_ens_pq - mean_unet_pq

    print("\n" + "=" * 65, flush=True)
    print("STAGE 2 ENSEMBLE RESULTS:", flush=True)
    print("=" * 65, flush=True)
    print(f"U-Net alone   PQ: {mean_unet_pq:.4f}  TP={unet_tp} FP={unet_fp} FN={unet_fn}", flush=True)
    print(f"Ensemble      PQ: {mean_ens_pq:.4f}  TP={ens_tp} FP={ens_fp} FN={ens_fn}", flush=True)
    print(f"Delta              : {delta:+.4f}", flush=True)
    print(f"Gated detections added: {gated_added_total}", flush=True)
    print("-" * 65, flush=True)

    if delta >= 0.005:
        verdict = (
            f"IMPROVEMENT (PQ {mean_ens_pq:.4f} vs {mean_unet_pq:.4f}, delta={delta:+.4f}). "
            f"Generate test submission!"
        )
    else:
        verdict = (
            f"NO IMPROVEMENT (delta={delta:+.4f} < 0.005). "
            f"Stage 2 DEAD. Keep U-Net-only submission."
        )

    print(f"VERDICT: {verdict}", flush=True)
    print("=" * 65, flush=True)

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    result = {
        "unet_pq": mean_unet_pq,
        "ensemble_pq": mean_ens_pq,
        "delta": delta,
        "gated_added": gated_added_total,
        "gating": {"score_thresh": mrcnn_score_thresh, "iou_gate": mrcnn_iou_gate},
        "verdict": verdict,
        "unet_stats": {"tp": unet_tp, "fp": unet_fp, "fn": unet_fn},
        "ensemble_stats": {"tp": ens_tp, "fp": ens_fp, "fn": ens_fn},
    }
    with open(output_report, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_report}", flush=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: Strict-Gated Ensemble Evaluation")
    parser.add_argument("--score_thresh", type=float, default=0.85,
                        help="Minimum Mask R-CNN confidence score to include (default: 0.85)")
    parser.add_argument("--iou_gate", type=float, default=0.10,
                        help="Max IoU with any U-Net mask to be included (default: 0.10)")
    args = parser.parse_args()

    print(f"Stage 2 Strict-Gated Ensemble", flush=True)
    print(f"  score_thresh : {args.score_thresh}", flush=True)
    print(f"  iou_gate     : {args.iou_gate}", flush=True)
    run_stage2_ensemble(
        mrcnn_score_thresh=args.score_thresh,
        mrcnn_iou_gate=args.iou_gate,
    )
