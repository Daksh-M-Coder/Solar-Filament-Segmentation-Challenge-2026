import os
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image, encode_mask_to_rle
from src.unet_submit import verify_no_overlaps_in_instances, verify_submission_csv_overlaps

def generate_stage0_submissions(
    config_path: str = "configs/unet_config.yaml",
    test_img_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/test/test_images",
    output_dir: str = "submissions",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "production" not in cfg:
        raise ValueError("Production section not found in configs/unet_config.yaml! Run stage0_sweep first.")

    prod = cfg["production"]
    best_th = float(prod["score_threshold"])
    peak_min_dist = int(prod["peak_min_distance"])
    prune_len = int(prod["prune_threshold"])
    kernel_size = int(prod["kernel_size"])
    dist_tf = str(prod["dist_transform"])
    weights_path = cfg["paths"].get("best_model_path", "models/unet_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading U-Net from: {weights_path} | Device: {device}")
    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    test_files = sorted(glob.glob(os.path.join(test_img_dir, "*.jpeg")) + glob.glob(os.path.join(test_img_dir, "*.jpg")))
    print(f"Found {len(test_files)} test images.")

    thresholds = [
        round(best_th, 2),
        round(best_th + 0.05, 2),
        round(best_th - 0.05, 2),
    ]

    postproc_kwargs = {
        "peak_min_dist": peak_min_dist,
        "skel_prune_len": prune_len,
        "morph_kernel_size": kernel_size,
        "dist_transform_type": dist_tf,
    }

    print(f"Production Post-Processing Config: {postproc_kwargs}")
    print(f"Thresholds to generate: {thresholds}")

    os.makedirs(output_dir, exist_ok=True)
    generated_csvs = []

    for th in thresholds:
        out_filename = f"submission_unet_stage0_th{th:.2f}.csv"
        out_path = os.path.join(output_dir, out_filename)
        print(f"\n============================================================")
        print(f"Generating: {out_path} (Threshold: {th:.2f})")
        print(f"============================================================")

        results = []
        for p in tqdm(test_files, desc=f"Thresh {th:.2f}"):
            stem = os.path.splitext(os.path.basename(p))[0]
            instances = predict_single_image(
                model=model,
                img_path=p,
                device=device,
                thresh=th,
                use_tta=False,
                postproc_kwargs=postproc_kwargs,
            )

            # Task 0.6 Sanity check #1: Verify no overlapping masks locally
            verify_no_overlaps_in_instances(instances, img_name=stem)

            if len(instances) == 0:
                continue

            for inst_idx, inst_mask in enumerate(instances, start=1):
                rle_code = encode_mask_to_rle(inst_mask)
                results.append({
                    "filament_id": f"{stem}_{inst_idx}",
                    "segmentation_rle": rle_code
                })

        sub_df = pd.DataFrame(results)
        if sub_df.empty:
            sub_df = pd.DataFrame(columns=["filament_id", "segmentation_rle"])

        # Save without index, no extra columns
        sub_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(sub_df)} instances, {len(sub_df['filament_id'].apply(lambda x: x.split('_')[0]).unique())} images with masks)")

        # Task 0.6 Sanity check #2: Verify zero overlaps across the saved CSV
        verify_submission_csv_overlaps(out_path)
        generated_csvs.append(out_path)

    print("\nAll Stage 0 submissions generated and verified successfully!")

if __name__ == "__main__":
    generate_stage0_submissions()
