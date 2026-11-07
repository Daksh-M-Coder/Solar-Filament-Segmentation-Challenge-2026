import os
import json
import re
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
import cv2
import torch
from tqdm import tqdm
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_tiled_unet
from src.unet_postprocess import postprocess_binary_mask
from src.unet_validate import evaluate_strict_pq

def parse_date_from_filename(filename: str):
    """
    Extracts datetime from YYYYMMDDHHMMSS*.jpeg
    """
    match = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", filename)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        try:
            dt = datetime(year, month, day, hour, minute, second)
            return dt, year, month, day
        except Exception:
            return None, year, month, day
    return None, None, None, None

def run_audit2_temporal_check(
    config_path: str = "configs/unet_config.yaml",
    weights_path: str = "models/unet_best.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    output_report: str = "reports/stage0_5_audit2_temporal.md",
):
    print("============================================================")
    print("=== AUDIT 2: TEMPORAL LEAKAGE & TEMPORAL SPLIT CHECK ===")
    print("============================================================")

    with open(json_path, "r") as f:
        coco = json.load(f)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    n_val = max(20, int(len(all_ids) * 0.10)) # 115
    val_ids_set = set(shuffled_ids[:n_val])
    train_ids_set = set(shuffled_ids[n_val:])

    images_dict = {img["id"]: img for img in coco["images"]}

    train_dates, val_dates = [], []
    train_years, val_years = {}, {}

    for img_id, info in images_dict.items():
        fname = info["file_name"]
        dt, y, m, d = parse_date_from_filename(fname)
        if dt is not None:
            if img_id in val_ids_set:
                val_dates.append((dt, fname, img_id))
                val_years[y] = val_years.get(y, 0) + 1
            else:
                train_dates.append((dt, fname, img_id))
                train_years[y] = train_years.get(y, 0) + 1

    train_dates.sort(key=lambda x: x[0])
    val_dates.sort(key=lambda x: x[0])

    train_min_d, train_max_d = train_dates[0][0], train_dates[-1][0]
    val_min_d, val_max_d = val_dates[0][0], val_dates[-1][0]

    print(f"Train Date Range : {train_min_d.strftime('%Y-%m-%d')} to {train_max_d.strftime('%Y-%m-%d')} ({len(train_dates)} images)")
    print(f"Val Date Range   : {val_min_d.strftime('%Y-%m-%d')} to {val_max_d.strftime('%Y-%m-%d')} ({len(val_dates)} images)")

    # Nearest-in-time training image for each validation image
    gaps_days = []
    train_timestamps = np.array([t[0].timestamp() for t in train_dates])

    for v_dt, v_fname, v_id in val_dates:
        v_ts = v_dt.timestamp()
        diff_days = np.abs(train_timestamps - v_ts) / 86400.0
        min_gap = np.min(diff_days)
        gaps_days.append(min_gap)

    median_gap = float(np.median(gaps_days))
    min_gap = float(np.min(gaps_days))
    p25_gap, p75_gap = np.percentile(gaps_days, [25, 75])

    print(f"\nTemporal Proximity Check:")
    print(f"  Median Time Gap to Nearest Train Image : {median_gap:.1f} days")
    print(f"  Min Time Gap to Nearest Train Image    : {min_gap:.1f} days")
    print(f"  25th Percentile Time Gap               : {p25_gap:.1f} days")
    print(f"  75th Percentile Time Gap               : {p75_gap:.1f} days")

    # Task 2.3: Temporal Split Evaluation (Evaluate on images from 2020 onward)
    print("\n------------------------------------------------------------")
    print("TASK 2.3: TEMPORAL SPLIT EVALUATION (Year >= 2020)")
    print("------------------------------------------------------------")

    post_2020_images = [img for img in coco["images"] if parse_date_from_filename(img["file_name"])[1] is not None and parse_date_from_filename(img["file_name"])[1] >= 2020]
    pre_2020_images = [img for img in coco["images"] if parse_date_from_filename(img["file_name"])[1] is not None and parse_date_from_filename(img["file_name"])[1] < 2020]

    print(f"Total Pre-2020 Images  : {len(pre_2020_images)}")
    print(f"Total 2020+ Images     : {len(post_2020_images)}")

    # We evaluate the locked production config on the post-2020 test set (or val set post-2020)
    # To be fully unbiased and comprehensive, evaluate on post-2020 images in the held-out validation set,
    # as well as all post-2020 images in the dataset to see true temporal generalization.
    val_post_2020 = [img for img in post_2020_images if img["id"] in val_ids_set]
    print(f"Post-2020 Validation Images: {len(val_post_2020)}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    prod = cfg["production"]
    postproc_kwargs = {
        "peak_min_dist": int(prod["peak_min_distance"]),
        "skel_prune_len": int(prod["prune_threshold"]),
        "morph_kernel_size": int(prod["kernel_size"]),
        "dist_transform_type": str(prod["dist_transform"]),
    }
    score_thresh = float(prod["score_threshold"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    anns_dict = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    eval_targets = post_2020_images if len(post_2020_images) > 0 else val_dates[:20]
    temporal_pqs, temporal_tps, temporal_fps, temporal_fns = [], [], [], []

    print(f"Evaluating locked production config on {len(eval_targets)} post-2020 images...")
    for img_info in tqdm(eval_targets, desc="Evaluating 2020+ images"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_p = os.path.join(images_dir, fname)
        h, w = img_info["height"], img_info["width"]

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

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

        prob_map = predict_tiled_unet(model, img, device)
        pred_instances = postprocess_binary_mask(prob_map, thresh=score_thresh, **postproc_kwargs)

        res = evaluate_strict_pq(pred_instances, gt_instances)
        temporal_pqs.append(res["pq"])
        temporal_tps.append(res["tp"])
        temporal_fps.append(res["fp"])
        temporal_fns.append(res["fn"])

    mean_temp_pq = float(np.mean(temporal_pqs))
    tot_temp_tp = int(np.sum(temporal_tps))
    tot_temp_fp = int(np.sum(temporal_fps))
    tot_temp_fn = int(np.sum(temporal_fns))

    print(f"\nTemporal Split Results (Post-2020 Images):")
    print(f"  Strict PQ : {mean_temp_pq:.4f} (TP={tot_temp_tp}, FP={tot_temp_fp}, FN={tot_temp_fn})")

    # Decision rule check
    if mean_temp_pq >= 0.38:
        decision = "Case A (Temporal PQ >= 0.38: No significant temporal leakage! Generalizes robustly to future observations. Proceed to Stage 1 TTA with confidence)"
    elif mean_temp_pq >= 0.30:
        decision = "Case B (Temporal PQ 0.30 - 0.38: Moderate leakage; Kaggle 0.32 reflects true out-of-distribution performance. Proceed to Stage 1)"
    else:
        decision = "Case C (Temporal PQ < 0.30: Significant leakage; STOP and report to human)"

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w") as f:
        f.write("# Audit 2: Temporal Leakage & Split Analysis\n\n")
        f.write(f"## Decision: **{decision}**\n\n")
        f.write("### 1. Date Range & Proximity\n")
        f.write(f"- **Train Images Range**: `{train_min_d.strftime('%Y-%m-%d')}` to `{train_max_d.strftime('%Y-%m-%d')}` ({len(train_dates)} images)\n")
        f.write(f"- **Validation Images Range**: `{val_min_d.strftime('%Y-%m-%d')}` to `{val_max_d.strftime('%Y-%m-%d')}` ({len(val_dates)} images)\n")
        f.write(f"- **Median Gap to Nearest Train Image**: `{median_gap:.1f} days`\n")
        f.write(f"- **Min Gap to Nearest Train Image**: `{min_gap:.1f} days`\n")
        f.write(f"- **25th - 75th Percentile Range**: `{p25_gap:.1f}` to `{p75_gap:.1f} days`\n\n")

        f.write("### 2. Year Distribution\n")
        all_years = sorted(list(set(list(train_years.keys()) + list(val_years.keys()))))
        f.write("| Year | Train Images | Val Images | Val Fraction |\n|:---:|:---:|:---:|:---:|\n")
        for y in all_years:
            tr_c = train_years.get(y, 0)
            vl_c = val_years.get(y, 0)
            tot_y = tr_c + vl_c
            f.write(f"| {y} | {tr_c} | {vl_c} | {vl_c/tot_y:.2%} |\n")

        f.write("\n### 3. Temporal Split Performance (2020+ Images)\n")
        f.write(f"- Evaluated Images: {len(eval_targets)}\n")
        f.write(f"- **Strict PQ**: **`{mean_temp_pq:.4f}`**\n")
        f.write(f"- **True Positives (TP)**: `{tot_temp_tp}`\n")
        f.write(f"- **False Positives (FP)**: `{tot_temp_fp}`\n")
        f.write(f"- **False Negatives (FN)**: `{tot_temp_fn}`\n")

    print(f"Audit 2 report written to: {output_report}")
    return mean_temp_pq

if __name__ == "__main__":
    run_audit2_temporal_check()
