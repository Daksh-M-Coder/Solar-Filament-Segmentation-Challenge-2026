"""
Hard-Negative Mining Diagnostic:
1. Runs unet_cldice_v2_best.pth over the training set (1,039 images) at the locked config:
   - TTA = prob_avg (4-view flip probability averaging)
   - Locked postproc: peak_min_dist=55, skel_prune_len=15, morph_kernel_size=3, dist_transform=DIST_L1, thresh=0.45, min_area=300
2. Extracts predicted instances with:
   - Mean instance probability / confidence > 0.60
   - Best IoU == 0.000 against EVERY ground-truth filament on that image
3. Saves the top 500 by (area * confidence) as 512x512 crops around their bounding box with an all-zeros foreground mask into data/hard_negatives_v1/
4. Reports:
   - Total count of such hard negative instances
   - Number of unique training images they originated from
"""

import os
import json
import time
import argparse
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import get_tile_avg_prob, predict_single_tile_views
from src.unet_postprocess import resolve_overlaps

def compute_raw_stitched_probs_and_instances(
    model: torch.nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
    mean: float = 0.449,
    std: float = 0.226,
    thresh: float = 0.45,
    postproc_kwargs: dict = None,
):
    """
    Runs the exact locked TTA=prob_avg tiled inference to get both:
      1. The final stitched probability map [H, W] (to calculate instance confidence)
      2. The final post-processed instances (using the exact live locked path)
    """
    H, W = image_np.shape[:2]
    step = tile_size - overlap

    y_steps = list(range(0, H - tile_size + 1, step))
    if y_steps[-1] != H - tile_size:
        y_steps.append(H - tile_size)

    x_steps = list(range(0, W - tile_size + 1, step))
    if x_steps[-1] != W - tile_size:
        x_steps.append(W - tile_size)

    kwargs = postproc_kwargs or {}
    model.eval()

    vote_canvas = np.zeros((H, W), dtype=np.float32)
    coverage_canvas = np.zeros((H, W), dtype=np.float32)
    prob_canvas = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y1 in y_steps:
            for x1 in x_steps:
                y2, x2 = y1 + tile_size, x1 + tile_size
                crop = image_np[y1:y2, x1:x2]

                # 4-view prob avg for tile
                tile_avg_prob, _, _ = get_tile_avg_prob(
                    model=model, crop=crop, device=device, mean=mean, std=std
                )
                prob_canvas[y1:y2, x1:x2] += tile_avg_prob

                # Predict single tile views with prob_avg
                tile_mask, _, _ = predict_single_tile_views(
                    model=model,
                    crop=crop,
                    device=device,
                    mean=mean,
                    std=std,
                    thresh=thresh,
                    tta_mode="prob_avg",
                    postproc_kwargs=kwargs,
                )
                vote_canvas[y1:y2, x1:x2] += tile_mask
                coverage_canvas[y1:y2, x1:x2] += 1.0

    stitched_prob = prob_canvas / np.maximum(coverage_canvas, 1.0)
    stitched_mask = ((vote_canvas / np.maximum(coverage_canvas, 1.0)) >= 0.50).astype(np.uint8)

    from skimage.measure import label, regionprops
    labeled = label(stitched_mask, connectivity=2)
    regions = regionprops(labeled)
    raw_instances = []
    min_area = kwargs.get("min_area", 300)
    for r in regions:
        if r.area >= min_area:
            inst_m = np.zeros((H, W), dtype=np.uint8)
            inst_m[r.bbox[0]:r.bbox[2], r.bbox[1]:r.bbox[3]] = r.image.astype(np.uint8)
            raw_instances.append(inst_m)

    instances = resolve_overlaps(raw_instances, min_area=min_area)
    return instances, stitched_prob

def compute_iou(m1, m2):
    inter = np.logical_and(m1 > 0, m2 > 0).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(m1 > 0, m2 > 0).sum()
    return inter / union if union > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Extract Hard Negative Tiles from Training Set")
    parser.add_argument("--checkpoint", type=str, default="models/unet_cldice_v2_best.pth")
    parser.add_argument("--out_dir", type=str, default="data/hard_negatives_v1")
    parser.add_argument("--conf_thresh", type=float, default=0.60)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--max_images", type=int, default=1039, help="Max training images to evaluate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print("=== EXTRACTING HARD NEGATIVES FROM TRAINING SET ===")
    print("============================================================")
    print(f"Device        : {device}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Conf Thresh   : > {args.conf_thresh}")
    print(f"Max Top-K     : {args.top_k}")
    print(f"Output Dir    : {args.out_dir}")
    print(f"Crop Size     : {args.crop_size}x{args.crop_size}")
    print("============================================================", flush=True)

    json_path = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    images_dir = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    anns_dict = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    n_val = max(20, int(len(all_ids) * 0.10))
    train_ids = shuffled_ids[n_val:][:args.max_images] # 1,039 training images

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    postproc_kwargs = {
        "peak_min_dist": 55,
        "skel_prune_len": 15,
        "morph_kernel_size": 3,
        "dist_transform_type": "DIST_L1",
        "min_area": 300,
    }

    hard_negative_candidates = []
    unique_images_with_hn = set()
    total_hn_instances = 0

    start_time = time.time()

    for idx, img_id in enumerate(train_ids, 1):
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        img_arr = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img_arr is None:
            continue

        h, w = info["height"], info["width"]

        # Decode ground truth masks for this image
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

        # Run inference
        pred_insts, stitched_prob = compute_raw_stitched_probs_and_instances(
            model=model,
            image_np=img_arr,
            device=device,
            thresh=0.45,
            postproc_kwargs=postproc_kwargs,
        )

        img_had_hn = False

        for p_inst in pred_insts:
            # Check IoU against all ground truth
            best_iou = 0.0
            for g in gt_instances:
                iou = compute_iou(p_inst, g)
                if iou > best_iou:
                    best_iou = iou
                    if best_iou > 0.0:
                        break # Only zero IoU instances qualify

            if best_iou == 0.0:
                # Compute mean confidence inside the predicted instance
                inst_pixels = (p_inst > 0)
                mean_conf = float(np.mean(stitched_prob[inst_pixels]))
                area = int(np.sum(inst_pixels))

                if mean_conf > args.conf_thresh:
                    total_hn_instances += 1
                    img_had_hn = True
                    score = area * mean_conf

                    # Bounding box of instance
                    rows = np.where(np.any(inst_pixels, axis=1))[0]
                    cols = np.where(np.any(inst_pixels, axis=0))[0]
                    cy = int((rows.min() + rows.max()) / 2.0)
                    cx = int((cols.min() + cols.max()) / 2.0)

                    # Compute 512x512 crop bounds around center
                    cs = args.crop_size
                    y1 = max(0, min(h - cs, cy - cs // 2))
                    x1 = max(0, min(w - cs, cx - cs // 2))
                    y2, x2 = y1 + cs, x1 + cs

                    crop_img = img_arr[y1:y2, x1:x2].copy()

                    hard_negative_candidates.append({
                        "score": score,
                        "area": area,
                        "mean_conf": mean_conf,
                        "img_id": img_id,
                        "file_name": info["file_name"],
                        "crop_img": crop_img,
                    })

        if img_had_hn:
            unique_images_with_hn.add(img_id)

        if idx % 50 == 0 or idx == len(train_ids):
            elapsed = time.time() - start_time
            rate = idx / elapsed
            rem_s = (len(train_ids) - idx) / max(rate, 1e-4)
            print(
                f"[{idx}/{len(train_ids)}] Elapsed: {elapsed:.1f}s | Est Rem: {rem_s/60:.1f}m | "
                f"Total HN instances: {total_hn_instances} from {len(unique_images_with_hn)} unique images",
                flush=True,
            )

    print("\n============================================================")
    print("=== HARD NEGATIVE MINING DIAGNOSTIC SUMMARY ===")
    print("============================================================")
    print(f"Total training images evaluated  : {len(train_ids)}")
    print(f"Total hard negative instances    : {total_hn_instances}")
    print(f"Unique images containing HN      : {len(unique_images_with_hn)}")
    print("============================================================")

    # Sort descending by area * confidence
    hard_negative_candidates.sort(key=lambda x: x["score"], reverse=True)
    selected_hn = hard_negative_candidates[:args.top_k]

    print(f"Saving top {len(selected_hn)} hard negative tiles to: {args.out_dir}/")
    zero_mask = np.zeros((args.crop_size, args.crop_size), dtype=np.uint8)

    metadata_list = []
    for rank, item in enumerate(selected_hn, 1):
        tile_name = f"hn_{rank:04d}_score{item['score']:.0f}_conf{item['mean_conf']:.2f}_area{item['area']}.png"
        mask_name = f"hn_{rank:04d}_score{item['score']:.0f}_conf{item['mean_conf']:.2f}_area{item['area']}_mask.png"

        tile_path = os.path.join(args.out_dir, tile_name)
        mask_path = os.path.join(args.out_dir, mask_name)

        cv2.imwrite(tile_path, item["crop_img"])
        cv2.imwrite(mask_path, zero_mask)

        metadata_list.append({
            "rank": rank,
            "tile_file": tile_name,
            "mask_file": mask_name,
            "score": item["score"],
            "area": item["area"],
            "confidence": item["mean_conf"],
            "source_image": item["file_name"],
            "image_id": item["img_id"],
        })

    with open(os.path.join(args.out_dir, "hard_negatives_manifest.json"), "w") as f:
        json.dump({
            "total_hard_negatives_found": total_hn_instances,
            "unique_images_count": len(unique_images_with_hn),
            "saved_tiles_count": len(selected_hn),
            "manifest": metadata_list,
        }, f, indent=2)

    print(f"Saved manifest to: {os.path.join(args.out_dir, 'hard_negatives_manifest.json')}")

if __name__ == "__main__":
    main()
