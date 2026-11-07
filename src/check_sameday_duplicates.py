import os
import json
import re
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

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

def run_same_day_duplicate_check(
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    output_report: str = "reports/stage0_5_sameday_duplicates.md",
):
    print("============================================================")
    print("=== TASK: SAME-DAY DUPLICATE & SSIM LEAKAGE CHECK ===")
    print("============================================================")

    with open(json_path, "r") as f:
        coco = json.load(f)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    n_val = max(20, int(len(all_ids) * 0.10))  # 115 validation images
    val_ids_set = set(shuffled_ids[:n_val])
    train_ids_set = set(shuffled_ids[n_val:])

    images_dict = {img["id"]: img for img in coco["images"]}

    train_entries, val_entries = [], []

    for img_id, info in images_dict.items():
        fname = info["file_name"]
        dt, y, m, d = parse_date_from_filename(fname)
        if dt is not None:
            if img_id in val_ids_set:
                val_entries.append({"id": img_id, "file_name": fname, "dt": dt, "date_str": f"{y:04d}-{m:02d}-{d:02d}"})
            else:
                train_entries.append({"id": img_id, "file_name": fname, "dt": dt, "date_str": f"{y:04d}-{m:02d}-{d:02d}"})

    print(f"Total Train Images : {len(train_entries)}")
    print(f"Total Val Images   : {len(val_entries)}")

    # Step 1: Find validation images whose nearest training neighbor has a 0-day gap (< 1.0 day / same calendar date)
    train_ts = np.array([e["dt"].timestamp() for e in train_entries])
    
    zero_day_pairs = []

    for v in val_entries:
        v_ts = v["dt"].timestamp()
        diffs = np.abs(train_ts - v_ts) / 86400.0  # gap in days
        nearest_idx = int(np.argmin(diffs))
        min_gap_days = diffs[nearest_idx]
        
        # 0-day gap: either same calendar day or under 24 hours gap
        nearest_train = train_entries[nearest_idx]
        same_calendar_day = (v["date_str"] == nearest_train["date_str"])
        is_zero_day = (min_gap_days < 1.0) or same_calendar_day

        if is_zero_day:
            zero_day_pairs.append({
                "val_id": v["id"],
                "val_file": v["file_name"],
                "val_dt": v["dt"],
                "train_id": nearest_train["id"],
                "train_file": nearest_train["file_name"],
                "train_dt": nearest_train["dt"],
                "gap_hours": min_gap_days * 24.0,
                "gap_days": min_gap_days,
                "same_calendar_day": same_calendar_day,
            })

    print(f"Found {len(zero_day_pairs)} validation images with a 0-day gap to their nearest training image (out of {len(val_entries)} total val images).")

    # Step 2: Compute SSIM on 2048x2048 grayscale images
    results = []
    cat_near_dup = []   # SSIM > 0.95
    cat_marginal = []   # SSIM 0.80 - 0.95
    cat_different = []  # SSIM < 0.80

    for pair in tqdm(zero_day_pairs, desc="Computing SSIM"):
        v_path = os.path.join(images_dir, pair["val_file"])
        t_path = os.path.join(images_dir, pair["train_file"])

        v_img = cv2.imread(v_path, cv2.IMREAD_GRAYSCALE)
        t_img = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)

        if v_img is None or t_img is None:
            continue

        # Compute SSIM on 2048x2048
        val_ssim = float(ssim(v_img, t_img, data_range=255))
        pair["ssim"] = val_ssim

        if val_ssim > 0.95:
            pair["category"] = "Near-Duplicate (Real Leakage)"
            cat_near_dup.append(pair)
        elif val_ssim >= 0.80:
            pair["category"] = "Similar but Different (Marginal Leakage)"
            cat_marginal.append(pair)
        else:
            pair["category"] = "Clearly Different Observations (Not Leakage)"
            cat_different.append(pair)

        results.append(pair)

    # Sort descending by SSIM
    results.sort(key=lambda x: x["ssim"], reverse=True)

    all_ssims = [r["ssim"] for r in results]
    mean_ssim = float(np.mean(all_ssims)) if all_ssims else 0.0

    total_val = len(val_entries)
    n_near_dup = len(cat_near_dup)
    n_marginal = len(cat_marginal)
    n_different = len(cat_different)

    pct_near_dup = (n_near_dup / total_val) * 100.0

    # Decision rule
    if pct_near_dup < 10.0:
        decision = f"Pass: {pct_near_dup:.1f}% near-duplicates (< 10%). No significant leakage. Proceed to TTA."
    elif pct_near_dup <= 30.0:
        decision = f"Moderate Leakage: {pct_near_dup:.1f}% near-duplicates (10-30%). Note it. Proceed to TTA anyway."
    else:
        decision = f"Significant Leakage: {pct_near_dup:.1f}% near-duplicates (> 30%). STOP and report to human before TTA."

    print("\n============================================================")
    print("TASK SUMMARY & DECISION:")
    print("============================================================")
    print(f"Total Validation Images Evaluated : {total_val}")
    print(f"Pairs with 0-Day Gap Checked      : {len(results)}")
    print(f"Mean SSIM Across All Pairs        : {mean_ssim:.4f}")
    print(f"------------------------------------------------------------")
    print(f"1. SSIM > 0.95 (Near-Duplicate)   : {n_near_dup} ({pct_near_dup:.1f}% of val set)")
    print(f"2. SSIM 0.80-0.95 (Marginal)      : {n_marginal} ({(n_marginal/total_val)*100.0:.1f}% of val set)")
    print(f"3. SSIM < 0.80 (Clearly Different): {n_different} ({(n_different/total_val)*100.0:.1f}% of val set)")
    print(f"------------------------------------------------------------")
    print(f"Decision: {decision}")
    print("============================================================")

    # Write Markdown Report
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w") as f:
        f.write("# Same-Day Duplicate & SSIM Leakage Audit Report\n\n")
        f.write(f"## Decision: **{decision}**\n\n")
        
        f.write("### Executive Summary\n")
        f.write(f"- **Total Validation Images**: `{total_val}`\n")
        f.write(f"- **0-Day Gap Pairs Analyzed**: `{len(results)}`\n")
        f.write(f"- **Mean SSIM Across All Pairs**: `{mean_ssim:.4f}`\n\n")

        f.write("### Category Breakdown\n")
        f.write("| Category | SSIM Range | Count | % of Total Val Set | Leakage Severity |\n")
        f.write("|:---|:---:|:---:|:---:|:---|\n")
        f.write(f"| **Near-Duplicate** | `> 0.95` | **{n_near_dup}** | **{pct_near_dup:.1f}%** | Real Leakage |\n")
        f.write(f"| **Similar but Different** | `0.80 - 0.95` | **{n_marginal}** | **{(n_marginal/total_val)*100.0:.1f}%** | Marginal (Different observation hours) |\n")
        f.write(f"| **Clearly Different** | `< 0.80` | **{n_different}** | **{(n_different/total_val)*100.0:.1f}%** | Not Leakage |\n\n")

        f.write("### List of Near-Duplicate Validation Images (SSIM > 0.95)\n")
        if n_near_dup == 0:
            f.write("_No near-duplicate images detected (0 images with SSIM > 0.95)._\n\n")
        else:
            f.write("| # | Validation Image | Nearest Training Image | SSIM | Time Gap (Hours) |\n")
            f.write("|:---:|:---|:---|:---:|:---:|\n")
            for i, p in enumerate(cat_near_dup, start=1):
                f.write(f"| {i} | `{p['val_file']}` | `{p['train_file']}` | **{p['ssim']:.4f}** | {p['gap_hours']:.2f}h |\n")
            f.write("\n")

        f.write("### Detailed Pairwise Table (All 0-Day Validation Pairs)\n")
        f.write("| Validation Image | Nearest Training Image | Time Gap (h) | SSIM | Classification |\n")
        f.write("|:---|:---|:---:|:---:|:---|\n")
        for p in results:
            f.write(f"| `{p['val_file']}` | `{p['train_file']}` | {p['gap_hours']:.2f}h | {p['ssim']:.4f} | {p['category']} |\n")

    print(f"\nReport written to: {output_report}")
    return pct_near_dup

if __name__ == "__main__":
    run_same_day_duplicate_check()
