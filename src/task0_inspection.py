import os
import json
import glob
import pandas as pd
import numpy as np
import cv2

print("=== TASK 0.1: JSON INSPECTION ===")
json_path = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
if not os.path.exists(json_path):
    candidates = glob.glob("data/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    if candidates:
        json_path = candidates[0]
print(f"Loading JSON from: {json_path}")

with open(json_path, "r") as f:
    coco = json.load(f)

print(f"Total Images: {len(coco.get('images', []))}")
print(f"Total Annotations: {len(coco.get('annotations', []))}")
print(f"Categories: {coco.get('categories', [])}")

if len(coco.get("annotations", [])) > 0:
    first_ann = coco["annotations"][0]
    print(f"Keys of first annotation: {list(first_ann.keys())}")
    for k in ["spine", "skeleton", "centerline"]:
        has_k = k in first_ann and first_ann[k] is not None and len(first_ann[k]) > 0
        print(f"Key '{k}' present and non-empty: {has_k}")
        if has_k:
            val = first_ann[k]
            sample_val = val[:10] if isinstance(val, (list, str)) else val
            print(f"Sample '{k}' (len={len(val)}): {sample_val}")

print("\n=== TASK 0.2: SUBMISSION FORMAT INSPECTION ===")
# Look in archive or submissions directory
sub_candidates = glob.glob("archive/**/submissions/*.csv", recursive=True) + glob.glob("submissions/*.csv", recursive=True)
if sub_candidates:
    sub_path = sub_candidates[0]
    print(f"Loading sample submission from: {sub_path}")
    sub_df = pd.read_csv(sub_path)
    print(f"Columns: {list(sub_df.columns)}")
    print(f"Total Rows: {len(sub_df)}")
    print("First 3 rows:")
    print(sub_df.head(3).to_string())
    print("Sample filament_id:", sub_df["filament_id"].iloc[0])
    print("Sample segmentation_rle (first 40 chars):", str(sub_df["segmentation_rle"].iloc[0])[:40])
    print("RLE Type:", type(sub_df["segmentation_rle"].iloc[0]))
    print("Has Index column: False (standard CSV index=False)")
else:
    print("No sample submission found.")

print("\n=== TASK 0.3: IMAGE ARTIFACT INSPECTION (STEP B) ===")
os.makedirs("reports", exist_ok=True)
train_imgs_pattern = "data/**/train_images/*.jpeg"
img_files = glob.glob(train_imgs_pattern, recursive=True)
if not img_files:
    img_files = glob.glob("data/**/train_images/*.jpg", recursive=True)

print(f"Found {len(img_files)} train images. Sampling 25 for artifact check...")
np.random.seed(42)
sampled_files = np.random.choice(img_files, min(25, len(img_files)), replace=False)

flagged_positions = {}

for img_p in sampled_files:
    img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    h, w = img.shape
    
    rows_to_check = [h // 4, h // 2, (3 * h) // 4]
    cols_to_check = [w // 4, w // 2, (3 * w) // 4]
    
    for r in rows_to_check:
        sig = img[r, :].astype(np.float32)
        diff = np.abs(np.diff(sig))
        med = np.median(diff)
        mad = np.median(np.abs(diff - med))
        thresh = med + 5.0 * (mad if mad > 1e-4 else 1.0)
        peaks = np.where(diff > thresh)[0]
        for p in peaks:
            key = f"row_{r}_col_{p}"
            flagged_positions[key] = flagged_positions.get(key, 0) + 1

    for c in cols_to_check:
        sig = img[:, c].astype(np.float32)
        diff = np.abs(np.diff(sig))
        med = np.median(diff)
        mad = np.median(np.abs(diff - med))
        thresh = med + 5.0 * (mad if mad > 1e-4 else 1.0)
        peaks = np.where(diff > thresh)[0]
        for p in peaks:
            key = f"col_{c}_row_{p}"
            flagged_positions[key] = flagged_positions.get(key, 0) + 1

seams = {k: v for k, v in flagged_positions.items() if v > 3}
print(f"Repeat discontinuity positions across images (> 3 images): {len(seams)}")
with open("reports/artifact_findings.md", "w") as f:
    f.write("# Artifact Inspection Findings\n\n")
    f.write(f"- Sampled images: {len(sampled_files)}\n")
    f.write(f"- Seams / repeat discontinuity count (> 3 images): {len(seams)}\n")
    for k, v in sorted(seams.items(), key=lambda x: x[1], reverse=True)[:20]:
        f.write(f"  - {k}: seen in {v} images\n")

print("Artifact inspection findings written to reports/artifact_findings.md")
