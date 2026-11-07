import os
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image, encode_mask_to_rle
from src.unet_submit import verify_no_overlaps_in_instances, verify_submission_csv_overlaps

def run_stage1_inference(
    config_path: str = "configs/unet_config.yaml",
    test_img_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/test/test_images",
    output_dir: str = "submissions",
    use_tta: bool = True,
    tta_mode: str = "mask_vote",
    score_thresh: float = 0.45,
    custom_output: str = None,
    checkpoint: str = None,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    prod = cfg["production"]
    peak_min_dist = int(prod["peak_min_distance"])
    prune_len = int(prod["prune_threshold"])
    kernel_size = int(prod["kernel_size"])
    dist_tf = str(prod["dist_transform"])
    weights_path = checkpoint if checkpoint is not None else cfg["paths"].get("best_model_path", "models/unet_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print(f"=== STAGE 1: INFERENCE PIPELINE ===")
    print("============================================================")
    print(f"Device: {device} | Weights: {weights_path}")
    print(f"TTA Enabled: {use_tta} | TTA Mode: {tta_mode} | Score Threshold: {score_thresh}")
    print(f"Production Post-Processing: peak={peak_min_dist}, prune={prune_len}, kernel={kernel_size}, dt={dist_tf}")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    test_files = sorted(glob.glob(os.path.join(test_img_dir, "*.jpeg")) + glob.glob(os.path.join(test_img_dir, "*.jpg")))
    print(f"Found {len(test_files)} test images.")

    postproc_kwargs = {
        "peak_min_dist": peak_min_dist,
        "skel_prune_len": prune_len,
        "morph_kernel_size": kernel_size,
        "dist_transform_type": dist_tf,
    }

    if not use_tta:
        mode_tag = "noTTA"
    else:
        mode_tag = f"TTA_{tta_mode}"

    if custom_output:
        out_path = custom_output
    else:
        out_path = os.path.join(output_dir, f"submission_unet_stage1_{mode_tag}_th{score_thresh:.2f}.csv")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Target Output CSV: {out_path}\n")

    results = []
    total_unstable_tiles = 0
    total_tiles_evaluated = 0
    variances_all = []

    for p in tqdm(test_files, desc=f"Inference ({mode_tag}, th={score_thresh:.2f})"):
        stem = os.path.splitext(os.path.basename(p))[0]
        instances, unstable_cnt, tot_tiles, mean_var = predict_single_image(
            model=model,
            img_path=p,
            device=device,
            thresh=score_thresh,
            use_tta=use_tta,
            tta_mode=tta_mode,
            postproc_kwargs=postproc_kwargs,
        )

        total_unstable_tiles += unstable_cnt
        total_tiles_evaluated += tot_tiles
        variances_all.append(mean_var)

        # Task 7: Local Sanity Check 1 - Verify no overlapping masks
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

    # Task 7: Verify format matches sample_submission.csv exactly
    sub_df.to_csv(out_path, index=False)

    num_images_covered = len(sub_df['filament_id'].apply(lambda x: x.split('_')[0]).unique()) if not sub_df.empty else 0
    print(f"\nSaved Submission: {out_path}")
    print(f"  Total Rows / Instances : {len(sub_df)}")
    print(f"  Images with Filaments  : {num_images_covered} / {len(test_files)}")

    if use_tta and total_tiles_evaluated > 0:
        unstable_pct = (total_unstable_tiles / total_tiles_evaluated) * 100.0
        avg_var = float(np.mean(variances_all)) if variances_all else 0.0
        print(f"  Mean Raw Probability Variance : {avg_var:.4f}")
        print(f"  Unstable Tiles (> 0.05 raw var): {total_unstable_tiles} / {total_tiles_evaluated} ({unstable_pct:.2f}%)")

    # Task 7: Local Sanity Check 2 - Verify zero overlaps on written CSV
    verify_submission_csv_overlaps(out_path)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 TTA Inference Pipeline")
    parser.add_argument("--tta", dest="use_tta", action="store_true", help="Enable test-time augmentation (TTA)")
    parser.add_argument("--no-tta", dest="use_tta", action="store_false", help="Disable test-time augmentation")
    parser.set_defaults(use_tta=True)
    parser.add_argument("--tta_mode", type=str, choices=["mask_vote", "prob_avg"], default="mask_vote",
                        help="TTA mode: 'mask_vote' (default) or 'prob_avg' (raw probability averaging)")
    parser.add_argument("--score_thresh", type=float, default=0.45, help="Score binarization threshold")
    parser.add_argument("--config", default="configs/unet_config.yaml", help="Path to config YAML")
    parser.add_argument("--test_dir", default="data/MAGFiLO_1.0_Kaggle_2026/test/test_images", help="Test images directory")
    parser.add_argument("--output", default=None, help="Custom output CSV path")
    parser.add_argument("--checkpoint", default=None, help="Custom weights checkpoint path")
    args = parser.parse_args()

    run_stage1_inference(
        config_path=args.config,
        test_img_dir=args.test_dir,
        output_dir="submissions",
        use_tta=args.use_tta,
        tta_mode=args.tta_mode,
        score_thresh=args.score_thresh,
        custom_output=args.output,
        checkpoint=args.checkpoint,
    )
