"""
Diagnostic 2: Filament Area and Contrast Analysis.
For every ground-truth filament in the training set:
  1. Computes its mask area in pixels.
  2. Computes its mean |intensity - local background| contrast.
     (Local background = dilated mask region minus filament mask).
  3. Reports quartile boundaries for area and contrast.
  4. Saves filament IDs in bottom quartile by area and by contrast to JSON.
"""

import os
import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils

def main():
    json_path = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    images_dir = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    
    # Exclude validation images if we want pure training pool,
    # or process all training set annotations:
    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    n_val = max(20, int(len(all_ids) * 0.10))
    train_ids = set(shuffled_ids[n_val:]) # 1039 training images

    # Filter annotations to train_ids
    train_anns = [ann for ann in coco["annotations"] if ann["image_id"] in train_ids]
    print(f"Total training annotations to evaluate: {len(train_anns)} across {len(train_ids)} images", flush=True)

    # Group by image_id for efficient disk reading
    anns_by_img = {}
    for ann in train_anns:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    filament_records = []
    
    # Pre-define dilation kernel for local background (15px dilation around filament)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    img_count = 0
    total_imgs = len(anns_by_img)

    for img_id, anns in anns_by_img.items():
        img_count += 1
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        img_arr = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img_arr is None:
            continue

        h, w = info["height"], info["width"]

        for ann in anns:
            ann_id = ann["id"]
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(mask_utils.merge(rles))
            elif isinstance(seg, dict):
                m = mask_utils.decode(seg)
            else:
                continue

            area = int(np.sum(m > 0))
            if area == 0:
                continue

            # Crop around the filament with margin for speed
            rows = np.where(np.any(m > 0, axis=1))[0]
            cols = np.where(np.any(m > 0, axis=0))[0]
            margin = 25
            r0, r1 = max(0, rows.min() - margin), min(h, rows.max() + margin)
            c0, c1 = max(0, cols.min() - margin), min(w, cols.max() + margin)

            m_crop = m[r0:r1, c0:c1]
            img_crop = img_arr[r0:r1, c0:c1]

            # Filament pixels
            fg_pixels = img_crop[m_crop > 0]
            fg_mean = float(np.mean(fg_pixels))

            # Local background: dilate mask by 15px within the crop, subtract filament mask
            dilated = cv2.dilate(m_crop.astype(np.uint8), bg_kernel, iterations=1)
            bg_mask = (dilated > 0) & (m_crop == 0)

            if np.sum(bg_mask) > 0:
                bg_pixels = img_crop[bg_mask]
                bg_mean = float(np.mean(bg_pixels))
                contrast = abs(fg_mean - bg_mean)
            else:
                contrast = 0.0


            filament_records.append({
                "filament_id": ann_id,
                "image_id": img_id,
                "file_name": info["file_name"],
                "area": area,
                "contrast": contrast,
                "fg_mean": fg_mean,
            })

        if img_count % 100 == 0 or img_count == total_imgs:
            print(f"Processed [{img_count}/{total_imgs}] images ({len(filament_records)} filaments)...", flush=True)

    areas = np.array([r["area"] for r in filament_records])
    contrasts = np.array([r["contrast"] for r in filament_records])

    # Quartiles
    q25_area, q50_area, q75_area = np.percentile(areas, [25, 50, 75])
    q25_contrast, q50_contrast, q75_contrast = np.percentile(contrasts, [25, 50, 75])

    print("\n============================================================")
    print("=== FILAMENT AREA & CONTRAST QUARTILE REPORT ===")
    print("============================================================")
    print(f"Total filaments analyzed: {len(filament_records)}")
    print(f"Area Quartiles (pixels):")
    print(f"  Min  : {areas.min()}")
    print(f"  Q1   : {q25_area:.1f}")
    print(f"  Q2   : {q50_area:.1f} (Median)")
    print(f"  Q3   : {q75_area:.1f}")
    print(f"  Max  : {areas.max()}")
    print(f"\nContrast Quartiles (|intensity - local background|):")
    print(f"  Min  : {contrasts.min():.2f}")
    print(f"  Q1   : {q25_contrast:.2f}")
    print(f"  Q2   : {q50_contrast:.2f} (Median)")
    print(f"  Q3   : {q75_contrast:.2f}")
    print(f"  Max  : {contrasts.max():.2f}")
    print("============================================================")

    # Extract bottom quartiles
    bottom_q_area = [r["filament_id"] for r in filament_records if r["area"] <= q25_area]
    bottom_q_contrast = [r["filament_id"] for r in filament_records if r["contrast"] <= q25_contrast]
    hardest_filaments = [r["filament_id"] for r in filament_records if r["area"] <= q25_area and r["contrast"] <= q25_contrast]

    print(f"Bottom quartile by area count     : {len(bottom_q_area)}")
    print(f"Bottom quartile by contrast count : {len(bottom_q_contrast)}")
    print(f"Both (small AND faint) count      : {len(hardest_filaments)}")

    os.makedirs("data", exist_ok=True)
    out_json = "data/filament_oversampling_targets.json"
    with open(out_json, "w") as f:
        json.dump({
            "quartiles": {
                "area": {"min": float(areas.min()), "q1": float(q25_area), "q2": float(q50_area), "q3": float(q75_area), "max": float(areas.max())},
                "contrast": {"min": float(contrasts.min()), "q1": float(q25_contrast), "q2": float(q50_contrast), "q3": float(q75_contrast), "max": float(contrasts.max())},
            },
            "bottom_q_area_ids": bottom_q_area,
            "bottom_q_contrast_ids": bottom_q_contrast,
            "both_ids": hardest_filaments,
        }, f, indent=2)

    print(f"Saved oversampling targets to: {out_json}")

if __name__ == "__main__":
    main()
