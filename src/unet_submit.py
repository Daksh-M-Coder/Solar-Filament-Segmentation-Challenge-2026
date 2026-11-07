import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image, encode_mask_to_rle

def verify_no_overlaps_in_instances(instances: list, img_name: str = ""):
    """
    Asserts that across all instance masks for an image, no pixel is shared by more than 1 instance.
    """
    if len(instances) <= 1:
        return
    H, W = instances[0].shape
    canvas = np.zeros((H, W), dtype=np.int32)
    for idx, inst in enumerate(instances, 1):
        canvas += (inst > 0).astype(np.int32)
    max_overlap = int(np.max(canvas))
    if max_overlap > 1:
        num_overlapping_pixels = int(np.sum(canvas > 1))
        raise AssertionError(f"Overlap violation in {img_name}! {num_overlapping_pixels} pixels shared (max overlap: {max_overlap}).")

def verify_submission_csv_overlaps(csv_path: str, height: int = 2048, width: int = 2048):
    """
    Decodes all RLEs in a submission CSV per image and checks for any overlapping masks.
    """
    print(f"Verifying overlap integrity of {csv_path}...")
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Empty submission, no overlaps.")
        return

    # Group by image stem
    grouped = {}
    for _, row in df.iterrows():
        fid = str(row["filament_id"])
        # filament_id is stem_instIdx
        stem = "_".join(fid.split("_")[:-1]) if "_" in fid else fid
        rle = str(row["segmentation_rle"])
        grouped.setdefault(stem, []).append((fid, rle))

    total_violations = 0
    for stem, items in grouped.items():
        if len(items) <= 1:
            continue
        canvas = np.zeros((height, width), dtype=np.int32)
        for fid, rle_str in items:
            if not rle_str or pd.isna(rle_str):
                continue
            rle_obj = {"size": [height, width], "counts": rle_str.encode("utf-8")}
            m = mask_utils.decode(rle_obj)
            canvas += (m > 0).astype(np.int32)

        max_ov = int(np.max(canvas))
        if max_ov > 1:
            ov_px = int(np.sum(canvas > 1))
            print(f"FAILED OVERLAP CHECK on {stem}: {ov_px} overlapping pixels detected (max={max_ov})")
            total_violations += 1

    if total_violations == 0:
        print(f"SUCCESS: Zero overlapping masks across all {len(grouped)} images in {csv_path}!")
    else:
        raise AssertionError(f"Found {total_violations} images with overlapping masks in {csv_path}!")

def generate_unet_submissions(
    weights_path: str = "models/unet_best.pth",
    test_img_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/test/test_images",
    thresholds: list = [0.55],
    peak_min_distance: int = 30,
    output_dir: str = "submissions",
    use_tta: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading U-Net from: {weights_path} | Device: {device}")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    test_files = sorted(glob.glob(os.path.join(test_img_dir, "*.jpeg")) + glob.glob(os.path.join(test_img_dir, "*.jpg")))
    print(f"Found {len(test_files)} test images. Peak min distance: {peak_min_distance}")

    postproc_kwargs = {
        "peak_min_dist": peak_min_distance,
    }

    for th in thresholds:
        print(f"\n--- Generating predictions for threshold {th:.2f} (peak_min_dist={peak_min_distance}) ---")
        results = []

        for p in tqdm(test_files, desc=f"Thresh {th:.2f}"):
            stem = os.path.splitext(os.path.basename(p))[0]
            instances = predict_single_image(
                model=model,
                img_path=p,
                device=device,
                thresh=th,
                use_tta=use_tta,
                postproc_kwargs=postproc_kwargs,
            )

            # Local verification before adding
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

        out_path = os.path.join(output_dir, f"submission_unet_th{th:.2f}.csv")
        sub_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(sub_df)} instances)")

        # Verify output CSV
        verify_submission_csv_overlaps(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/unet_best.pth")
    parser.add_argument("--test_dir", default="data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.55])
    parser.add_argument("--peak_min_distance", type=int, default=30)
    parser.add_argument("--use_tta", action="store_true")
    args = parser.parse_args()

    generate_unet_submissions(
        weights_path=args.weights,
        test_img_dir=args.test_dir,
        thresholds=args.thresholds,
        peak_min_distance=args.peak_min_distance,
        use_tta=args.use_tta,
    )
