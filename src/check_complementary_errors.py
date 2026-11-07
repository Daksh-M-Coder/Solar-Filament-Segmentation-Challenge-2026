import os
import json
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image
from src.unet_postprocess import postprocess_binary_mask, resolve_overlaps
from src.audit_maskrcnn_115 import build_mask_rcnn, predict_tiled_maskrcnn

def compute_mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    intersection = np.logical_and(m1, m2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(m1, m2).sum()
    return float(intersection) / float(union)

def match_instances_to_gt(pred_instances: list, gt_instances: list, iou_thresh: float = 0.50):
    """
    Hungarian bipartite matching between predicted instances and ground truth instances.
    Returns:
      matched_gt_indices: set of GT indices successfully matched (TPs)
      matched_pred_indices: set of pred indices successfully matched
      tp: int
      fp: int
      fn: int
    """
    n_pred = len(pred_instances)
    n_gt = len(gt_instances)

    if n_pred == 0 or n_gt == 0:
        return set(), set(), 0, n_pred, n_gt

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, p in enumerate(pred_instances):
        for j, g in enumerate(gt_instances):
            iou_matrix[i, j] = compute_mask_iou(p, g)

    cost_matrix = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_preds = set()
    matched_gts = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_thresh:
            matched_preds.add(r)
            matched_gts.add(c)

    tp = len(matched_gts)
    fp = n_pred - tp
    fn = n_gt - tp

    return matched_gts, matched_preds, tp, fp, fn

def run_complementary_error_check(
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    unet_weights: str = "models/unet_best.pth",
    mrcnn_weights: str = "models/best_model.pth",
    cache_tta_dir: str = "data/cache_tta_val115",
    cache_mrcnn_dir: str = "data/cache_mrcnn_val115",
    output_report: str = "reports/stage1_6_task2_complementary.json",
):
    print("=================================================================")
    print("=== TASK 2: COMPLEMENTARY-ERROR CHECK (U-NET VS MASK R-CNN) ===")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load annotations and define the 115 held-out pool
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

    print(f"Held-out pool: {len(val_ids)} images")

    # Set up models if caching is needed
    os.makedirs(cache_mrcnn_dir, exist_ok=True)
    mrcnn_model = None
    unet_model = None

    # Baseline locked post-processing parameters
    unet_postproc_kwargs = {
        "peak_min_dist": 55,
        "skel_prune_len": 15,
        "morph_kernel_size": 3,
        "dist_transform_type": "DIST_L1",
        "min_area": 300,
    }
    unet_thresh = 0.45

    tot_gt_count = 0
    unet_tot_tp = 0
    unet_tot_fp = 0
    unet_tot_fn = 0

    mrcnn_tot_tp = 0
    mrcnn_tot_fp = 0
    mrcnn_tot_fn = 0

    tot_complementary = 0
    tot_double_counted = 0

    per_image_results = []

    print("\nEvaluating both models and computing filament matching intersections...")
    for img_id in tqdm(val_ids, desc="Complementary Error Matching"):
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        h, w = info["height"], info["width"]

        # Ground truth filaments
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

        n_gt = len(gt_instances)
        tot_gt_count += n_gt

        # 1. Get U-Net Predictions
        # Prefer cached probability map if available
        tta_cache_p = os.path.join(cache_tta_dir, f"{img_id}.npz")
        if os.path.exists(tta_cache_p):
            data = np.load(tta_cache_p)
            prob_map = data["prob"].astype(np.float32)
            # postprocess_binary_mask already calls resolve_overlaps internally
            unet_insts = postprocess_binary_mask(prob_map, thresh=unet_thresh, **unet_postproc_kwargs)
        else:
            if unet_model is None:
                unet_model = UNetResNet34(pretrained=False).to(device)
                unet_model.load_state_dict(torch.load(unet_weights, map_location=device))
                unet_model.eval()
            unet_insts, _, _, _ = predict_single_image(
                model=unet_model,
                img_path=img_p,
                device=device,
                thresh=unet_thresh,
                use_tta=True,
                tta_mode="prob_avg",
                postproc_kwargs=unet_postproc_kwargs,
            )

        # 2. Get Mask R-CNN Predictions
        mrcnn_cache_p = os.path.join(cache_mrcnn_dir, f"{img_id}.npy")
        if os.path.exists(mrcnn_cache_p):
            mrcnn_insts = list(np.load(mrcnn_cache_p, allow_pickle=True))
        else:
            if mrcnn_model is None:
                if not os.path.exists(mrcnn_weights):
                    alt = "archive/Mask R-CNN/models/best_model.pth"
                    if os.path.exists(alt):
                        mrcnn_weights = alt
                mrcnn_model = build_mask_rcnn(num_classes=2, box_nms_thresh=0.5)
                ckpt = torch.load(mrcnn_weights, map_location=device)
                if "model_state_dict" in ckpt:
                    mrcnn_model.load_state_dict(ckpt["model_state_dict"])
                else:
                    mrcnn_model.load_state_dict(ckpt)
                mrcnn_model.to(device)
                mrcnn_model.eval()

            img_bgr = cv2.imread(img_p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            mrcnn_insts = predict_tiled_maskrcnn(
                mrcnn_model, img_tensor, device,
                tile_size=512, overlap=64,
                score_thresh=0.50, mask_thresh=0.50, nms_thresh=0.50
            )
            # Cache compressed boolean masks
            bool_masks = [(m > 0).astype(np.uint8) for m in mrcnn_insts]
            np.save(mrcnn_cache_p, np.array(bool_masks, dtype=object), allow_pickle=True)

        # 3. Match each model against GT
        u_matched_gts, _, u_tp, u_fp, u_fn = match_instances_to_gt(unet_insts, gt_instances, iou_thresh=0.50)
        m_matched_gts, _, m_tp, m_fp, m_fn = match_instances_to_gt(mrcnn_insts, gt_instances, iou_thresh=0.50)

        unet_tot_tp += u_tp
        unet_tot_fp += u_fp
        unet_tot_fn += u_fn

        mrcnn_tot_tp += m_tp
        mrcnn_tot_fp += m_fp
        mrcnn_tot_fn += m_fn

        # 4. Sets of ground truth filaments
        all_gt_indices = set(range(n_gt))
        u_fn_gts = all_gt_indices - u_matched_gts   # Ground truth filaments missed by U-Net
        m_tp_gts = m_matched_gts                    # Ground truth filaments caught by Mask R-CNN

        # Complementary: U-Net missed, Mask R-CNN caught
        comp_gts = u_fn_gts & m_tp_gts
        # Double-counting: caught by both
        overlap_gts = u_matched_gts & m_tp_gts

        tot_complementary += len(comp_gts)
        tot_double_counted += len(overlap_gts)

        per_image_results.append({
            "image_id": img_id,
            "n_gt": n_gt,
            "unet_tp": u_tp, "unet_fp": u_fp, "unet_fn": u_fn,
            "mrcnn_tp": m_tp, "mrcnn_fp": m_fp, "mrcnn_fn": m_fn,
            "complementary_count": len(comp_gts),
            "double_counted_count": len(overlap_gts),
        })

    recovery_pct = (tot_complementary / unet_tot_fn * 100.0) if unet_tot_fn > 0 else 0.0
    mrcnn_overlap_pct = (tot_double_counted / mrcnn_tot_tp * 100.0) if mrcnn_tot_tp > 0 else 0.0

    print("\n" + "=" * 65)
    print("TASK 2 RESULTS: COMPLEMENTARY-ERROR CHECK")
    print("=" * 65)
    print(f"Total Ground Truth Filaments          : {tot_gt_count}")
    print(f"U-Net TPs                              : {unet_tot_tp}")
    print(f"U-Net FPs                              : {unet_tot_fp}")
    print(f"U-Net FNs                              : {unet_tot_fn}")
    print("-" * 65)
    print(f"Mask R-CNN TPs                         : {mrcnn_tot_tp}")
    print(f"Mask R-CNN FPs                         : {mrcnn_tot_fp}")
    print(f"Mask R-CNN FNs                         : {mrcnn_tot_fn}")
    print("-" * 65)
    print(f"Complementary Filaments (U-Net missed, Mask R-CNN caught): {tot_complementary}")
    print(f"Percentage of U-Net FNs Recoverable   : {recovery_pct:.2f}% ({tot_complementary}/{unet_tot_fn})")
    print(f"Mask R-CNN TPs Already Caught by U-Net: {tot_double_counted} ({mrcnn_overlap_pct:.2f}% of MRC-NN TPs)")
    print("-" * 65)

    if tot_complementary > 40:
        verdict = "STAGE 2 VIABLE (Count > 40: worth trying with strict confidence gating)"
    elif tot_complementary >= 15:
        verdict = "MARGINAL (Count 15-40: test with very strict gating only)"
    else:
        verdict = "STAGE 2 IS DEAD (Count < 15: drop ensemble completely)"

    print(f"DECISION RULE VERDICT: {verdict}")
    print("=" * 65)

    summary_data = {
        "total_gt": tot_gt_count,
        "unet": {"tp": unet_tot_tp, "fp": unet_tot_fp, "fn": unet_tot_fn},
        "mrcnn": {"tp": mrcnn_tot_tp, "fp": mrcnn_tot_fp, "fn": mrcnn_tot_fn},
        "complementary_count": tot_complementary,
        "recovery_pct": recovery_pct,
        "double_counted_count": tot_double_counted,
        "double_counted_pct": mrcnn_overlap_pct,
        "verdict": verdict,
        "per_image": per_image_results,
    }

    with open(output_report, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Results saved to: {output_report}")
    return summary_data

if __name__ == "__main__":
    run_complementary_error_check()
