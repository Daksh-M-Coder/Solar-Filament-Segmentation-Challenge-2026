import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.unet_model import UNetResNet34
from src.unet_inference import get_tile_4views_raw_probs

def compute_image_tta_prob_map(
    model: torch.nn.Module,
    img_gray: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
    mean: float = 0.449,
    std: float = 0.226,
) -> np.ndarray:
    """
    Computes a full 2048x2048 TTA-averaged probability map by:
    1. For each tile (512x512 with 64 overlap):
       - Running 4-view TTA (identity, hflip, vflip, hvflip)
       - Un-flipping and averaging the 4 probability maps
    2. Accumulating tile probabilities with overlap blending
    Returns float32 probability map in [0, 1].
    """
    H, W = img_gray.shape[:2]
    step = tile_size - overlap

    y_steps = list(range(0, H - tile_size + 1, step))
    if y_steps[-1] != H - tile_size:
        y_steps.append(H - tile_size)

    x_steps = list(range(0, W - tile_size + 1, step))
    if x_steps[-1] != W - tile_size:
        x_steps.append(W - tile_size)

    prob_canvas = np.zeros((H, W), dtype=np.float32)
    count_canvas = np.zeros((H, W), dtype=np.float32)

    for y1 in y_steps:
        for x1 in x_steps:
            y2, x2 = y1 + tile_size, x1 + tile_size
            crop = img_gray[y1:y2, x1:x2]

            (p1, p2, p3, p4), _, _ = get_tile_4views_raw_probs(
                model=model, crop=crop, device=device, mean=mean, std=std
            )
            tile_avg = (p1 + p2 + p3 + p4) / 4.0

            prob_canvas[y1:y2, x1:x2] += tile_avg
            count_canvas[y1:y2, x1:x2] += 1.0

    full_tta_prob = prob_canvas / np.maximum(count_canvas, 1.0)
    return full_tta_prob

def cache_all_tta_maps(
    weights_path: str = "models/unet_best.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    output_dir: str = "data/cache_tta_val115",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== CACHING TTA-AVERAGED PROBABILITY MAPS (115 POOL) ===")
    print(f"Device: {device} | Model: {weights_path}")
    print(f"Output directory: {output_dir}")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    val_ids = shuffled_ids[:max(20, int(len(all_ids) * 0.10))]

    print(f"Target pool: {len(val_ids)} images")

    saved_count = 0
    skipped_count = 0

    for img_id in tqdm(val_ids, desc="Caching TTA maps"):
        cache_file = os.path.join(output_dir, f"{img_id}.npz")
        if os.path.exists(cache_file):
            skipped_count += 1
            continue

        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        img_gray = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        full_prob = compute_image_tta_prob_map(model, img_gray, device)
        # Store as float16 to save disk space (~8MB per image instead of 16MB)
        np.savez_compressed(cache_file, prob=full_prob.astype(np.float16))
        saved_count += 1

    print(f"Caching complete: {saved_count} saved, {skipped_count} already cached.")

if __name__ == "__main__":
    cache_all_tta_maps()
