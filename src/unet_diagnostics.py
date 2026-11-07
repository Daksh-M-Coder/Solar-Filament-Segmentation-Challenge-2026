import os
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_tiled_unet
from src.unet_postprocess import postprocess_binary_mask

def run_visual_diagnostics(
    weights_path: str = "models/unet_best.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    output_dir: str = "reports/figures",
    num_samples: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Task 7.2 Visual Diagnostics on {num_samples} images using {device}...")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    anns_dict = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    # Deterministic sample of images with filaments
    candidate_ids = [img_id for img_id, anns in anns_dict.items() if len(anns) > 0]
    np.random.seed(42)
    selected_ids = np.random.choice(candidate_ids, min(num_samples, len(candidate_ids)), replace=False)

    for i, img_id in enumerate(selected_ids, start=1):
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        h, w = info["height"], info["width"]

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # GT mask
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns_dict.get(img_id, []):
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(mask_utils.merge(rles))
                gt_mask = np.maximum(gt_mask, m)
            elif isinstance(seg, dict):
                m = mask_utils.decode(seg)
                gt_mask = np.maximum(gt_mask, m)

        prob_map = predict_tiled_unet(model, img, device)
        instances = postprocess_binary_mask(prob_map, thresh=0.50)
        pred_combined = np.zeros((h, w), dtype=np.uint8)
        for inst in instances:
            pred_combined = np.maximum(pred_combined, inst)

        # Create overlay: GT green, Pred red
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay[gt_mask > 0] = [0, 255, 0]        # Green for GT
        overlay[pred_combined > 0] = [255, 0, 0]   # Red for Pred
        # Overlap (TP) in yellow
        tp_overlap = (gt_mask > 0) & (pred_combined > 0)
        overlay[tp_overlap] = [255, 255, 0]

        # Plot 4-panel comparison
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"Original: {info['file_name']}")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        axes[2].imshow(pred_combined, cmap="gray")
        axes[2].set_title("Predicted Mask (0.50)")
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (G:GT, R:Pred, Y:TP)")
        axes[3].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"diag_sample_{i:02d}_{info['file_name']}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    print("Visual diagnostics complete!")

if __name__ == "__main__":
    run_visual_diagnostics()
