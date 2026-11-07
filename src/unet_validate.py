import os
import json
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize

from src.unet_postprocess import prune_skeleton, count_branch_points

def compute_mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    intersection = np.logical_and(m1, m2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(m1, m2).sum()
    return float(intersection) / float(union)

def evaluate_strict_pq(pred_instances: list, gt_instances: list, iou_thresh: float = 0.50):
    n_pred = len(pred_instances)
    n_gt = len(gt_instances)

    if n_pred == 0 and n_gt == 0:
        return {"pq": 1.0, "tp": 0, "fp": 0, "fn": 0, "sq": 1.0, "rq": 1.0}
    if n_pred == 0:
        return {"pq": 0.0, "tp": 0, "fp": 0, "fn": n_gt, "sq": 0.0, "rq": 0.0}
    if n_gt == 0:
        return {"pq": 0.0, "tp": 0, "fp": n_pred, "fn": 0, "sq": 0.0, "rq": 0.0}

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, p in enumerate(pred_instances):
        for j, g in enumerate(gt_instances):
            iou_matrix[i, j] = compute_mask_iou(p, g)

    cost_matrix = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp_ious = []
    tp = 0
    matched_preds = set()
    matched_gts = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_thresh:
            tp += 1
            tp_ious.append(iou_matrix[r, c])
            matched_preds.add(r)
            matched_gts.add(c)

    fp = n_pred - tp
    fn = n_gt - tp

    sq = float(np.mean(tp_ious)) if tp > 0 else 0.0
    rq = float(tp) / float(tp + 0.5 * fp + 0.5 * fn)
    pq = sq * rq

    return {"pq": pq, "tp": tp, "fp": fp, "fn": fn, "sq": sq, "rq": rq}

def validate_unet(
    model: torch.nn.Module,
    json_path: str,
    images_dir: str,
    val_image_ids: list,
    device: torch.device,
    thresh: float = 0.50,
    limit: int = 20,
    use_tta: bool = False,
    postproc_kwargs: dict = None,
    precomputed_prob_maps: dict = None,
):
    from src.unet_inference import predict_tiled_unet
    from src.unet_postprocess import postprocess_binary_mask

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    anns_dict = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    eval_ids = val_image_ids[:limit]
    all_pqs, all_tps, all_fps, all_fns = [], [], [], []
    inst_counts = []
    branch_counts = []

    model.eval()
    for idx, img_id in enumerate(eval_ids, start=1):
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        h, w = info["height"], info["width"]

        # Ground truth instances
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

        # Get or compute probability map
        if precomputed_prob_maps is not None and img_id in precomputed_prob_maps:
            prob_map = precomputed_prob_maps[img_id]
        else:
            img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            prob_map = predict_tiled_unet(model, img, device, use_tta=use_tta)
            if precomputed_prob_maps is not None:
                precomputed_prob_maps[img_id] = prob_map

        kwargs = postproc_kwargs or {}
        pred_instances = postprocess_binary_mask(prob_map, thresh=thresh, **kwargs)

        inst_counts.append(len(pred_instances))
        for inst in pred_instances:
            sk = skeletonize(inst > 0).astype(np.uint8)
            prune_len = kwargs.get("skel_prune_len", 10)
            if prune_len > 0:
                sk = prune_skeleton(sk, min_branch_len=prune_len)
            bp = count_branch_points(sk)
            branch_counts.append(bp)

        res = evaluate_strict_pq(pred_instances, gt_instances)
        all_pqs.append(res["pq"])
        all_tps.append(res["tp"])
        all_fps.append(res["fp"])
        all_fns.append(res["fn"])

    mean_pq = float(np.mean(all_pqs))
    total_tp = int(np.sum(all_tps))
    total_fp = int(np.sum(all_fps))
    total_fn = int(np.sum(all_fns))
    precision = float(total_tp) / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp) / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "mean_pq": mean_pq,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "mean_instances_per_image": float(np.mean(inst_counts)) if inst_counts else 0.0,
        "mean_branch_points": float(np.mean(branch_counts)) if branch_counts else 0.0,
    }
