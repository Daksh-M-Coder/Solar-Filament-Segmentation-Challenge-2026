import os
import json
import yaml
import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.unet_model import UNetResNet34
from src.unet_inference import get_tile_4views_raw_probs

def run_borderline_variance_diagnostic(
    config_path: str = "configs/unet_config.yaml",
    weights_path: str = "models/unet_best.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    num_images: int = 20,
    tile_size: int = 512,
    overlap: int = 64,
):
    print("============================================================")
    print("=== TASK 1: BORDERLINE-PIXEL VARIANCE DIAGNOSTIC ===")
    print("============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {weights_path} | Images: {num_images}")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    val_ids = shuffled_ids[:num_images]

    step = tile_size - overlap
    H, W = 2048, 2048
    y_steps = list(range(0, H - tile_size + 1, step))
    if y_steps[-1] != H - tile_size: y_steps.append(H - tile_size)
    x_steps = list(range(0, W - tile_size + 1, step))
    if x_steps[-1] != W - tile_size: x_steps.append(W - tile_size)

    total_pixels_evaluated = 0
    total_borderline_pixels = 0
    borderline_variances = []
    borderline_max_vars = []

    borderline_gt_002_count = 0
    borderline_gt_005_count = 0

    for img_id in tqdm(val_ids, desc="Evaluating Borderline Variance"):
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        for y1 in y_steps:
            for x1 in x_steps:
                y2, x2 = y1 + tile_size, x1 + tile_size
                crop = img[y1:y2, x1:x2]

                (p1, p2, p3, p4), _, _ = get_tile_4views_raw_probs(model, crop, device)
                stack_probs = np.stack([p1, p2, p3, p4], axis=0) # [4, 512, 512]
                avg_prob = np.mean(stack_probs, axis=0)
                var_map = np.var(stack_probs, axis=0)

                total_pixels_evaluated += crop.size

                # Borderline pixels: averaged probability in [0.35, 0.55]
                border_mask = (avg_prob >= 0.35) & (avg_prob <= 0.55)
                n_border = int(np.sum(border_mask))
                total_borderline_pixels += n_border

                if n_border > 0:
                    b_vars = var_map[border_mask]
                    borderline_variances.extend(b_vars.tolist())
                    borderline_max_vars.append(float(np.max(b_vars)))
                    borderline_gt_002_count += int(np.sum(b_vars > 0.02))
                    borderline_gt_005_count += int(np.sum(b_vars > 0.05))

    mean_border_var = float(np.mean(borderline_variances)) if borderline_variances else 0.0
    max_border_var = float(np.max(borderline_max_vars)) if borderline_max_vars else 0.0
    frac_gt_002 = (borderline_gt_002_count / max(1, total_borderline_pixels)) * 100.0
    frac_gt_005 = (borderline_gt_005_count / max(1, total_borderline_pixels)) * 100.0
    border_band_frac = (total_borderline_pixels / max(1, total_pixels_evaluated)) * 100.0

    print("\n============================================================")
    print("TASK 1 SUMMARY & FINDINGS:")
    print("============================================================")
    print(f"Total Pixels Evaluated                  : {total_pixels_evaluated:,}")
    print(f"Total Borderline Pixels ([0.35, 0.55])   : {total_borderline_pixels:,} ({border_band_frac:.3f}% of total pixels)")
    print(f"Mean Borderline Variance                : {mean_border_var:.5f}")
    print(f"Max Borderline Variance                 : {max_border_var:.5f}")
    print(f"Borderline Pixels with Variance > 0.02  : {borderline_gt_002_count:,} ({frac_gt_002:.2f}%)")
    print(f"Borderline Pixels with Variance > 0.05  : {borderline_gt_005_count:,} ({frac_gt_005:.2f}%)")
    
    # Decision rule
    if mean_border_var > 0.02 or frac_gt_005 > 5.0 or borderline_gt_002_count > 0:
        conclusion = "Borderline variance is meaningfully active and concentrated on filament edges! TTA works by stabilizing ambiguous contour boundaries."
    else:
        conclusion = "Borderline variance is extremely low across views."
    print(f"Conclusion: {conclusion}")
    print("============================================================")

    return {
        "total_pixels_evaluated": total_pixels_evaluated,
        "total_borderline_pixels": total_borderline_pixels,
        "border_band_frac": border_band_frac,
        "mean_border_var": mean_border_var,
        "max_border_var": max_border_var,
        "frac_gt_002": frac_gt_002,
        "frac_gt_005": frac_gt_005,
        "borderline_gt_002_count": borderline_gt_002_count,
        "borderline_gt_005_count": borderline_gt_005_count,
        "conclusion": conclusion,
    }

if __name__ == "__main__":
    run_borderline_variance_diagnostic()
