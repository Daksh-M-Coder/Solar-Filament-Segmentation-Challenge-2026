import os
import json
import time
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image
from src.unet_validate import evaluate_strict_pq

def run_stage1_tta_local_comparison(
    config_path: str = "configs/unet_config.yaml",
    weights_path: str = "models/unet_best.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    output_report: str = "reports/stage1_tta_comparison.md",
    score_thresh: float = 0.45,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    prod = cfg["production"]
    peak_min_dist = int(prod["peak_min_distance"])
    prune_len = int(prod["prune_threshold"])
    kernel_size = int(prod["kernel_size"])
    dist_tf = str(prod["dist_transform"])

    print("============================================================")
    print("=== TASK 5: STAGE 1 TTA LOCAL COMPARISON (115 IMAGES) ===")
    print("============================================================")
    print(f"Postproc Config: peak={peak_min_dist}, prune={prune_len}, kernel={kernel_size}, dt={dist_tf}, score={score_thresh}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {weights_path}")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    anns_dict = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    n_val = max(20, int(len(all_ids) * 0.10)) # 115 validation images
    val_ids = shuffled_ids[:n_val]

    print(f"Validation pool size: {len(val_ids)} images")

    postproc_kwargs = {
        "peak_min_dist": peak_min_dist,
        "skel_prune_len": prune_len,
        "morph_kernel_size": kernel_size,
        "dist_transform_type": dist_tf,
    }

    comparison_results = {}

    for mode in ["mask_vote", "prob_avg"]:
        print(f"\n------------------------------------------------------------")
        print(f"Evaluating Mode: {mode} (TTA: True, Thresh: {score_thresh})")
        print(f"------------------------------------------------------------")

        pqs, tps, fps, fns = [], [], [], []
        unstable_tiles_all = 0
        total_tiles_all = 0
        variances_all = []

        start_time = time.time()

        for img_id in tqdm(val_ids, desc=f"Evaluating {mode}"):
            info = images_dict[img_id]
            img_p = os.path.join(images_dir, info["file_name"])
            h, w = info["height"], info["width"]

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

            instances, unst_cnt, tot_cnt, mean_var = predict_single_image(
                model=model,
                img_path=img_p,
                device=device,
                thresh=score_thresh,
                use_tta=True,
                tta_mode=mode,
                postproc_kwargs=postproc_kwargs,
            )

            unstable_tiles_all += unst_cnt
            total_tiles_all += tot_cnt
            variances_all.append(mean_var)

            res = evaluate_strict_pq(instances, gt_instances)
            pqs.append(res["pq"])
            tps.append(res["tp"])
            fps.append(res["fp"])
            fns.append(res["fn"])

        elapsed_time = time.time() - start_time
        mean_pq = float(np.mean(pqs))
        tot_tp = int(np.sum(tps))
        tot_fp = int(np.sum(fps))
        tot_fn = int(np.sum(fns))
        mean_variance = float(np.mean(variances_all)) if variances_all else 0.0

        comparison_results[mode] = {
            "mean_pq": mean_pq,
            "tot_tp": tot_tp,
            "tot_fp": tot_fp,
            "tot_fn": tot_fn,
            "unstable_tiles": unstable_tiles_all,
            "total_tiles": total_tiles_all,
            "mean_variance": mean_variance,
            "elapsed_seconds": elapsed_time,
        }

        print(f"\n{mode} Results:")
        print(f"  Strict PQ        : {mean_pq:.4f}")
        print(f"  TP={tot_tp}, FP={tot_fp}, FN={tot_fn}")
        print(f"  Mean Raw Variance: {mean_variance:.4f}")
        print(f"  Unstable Tiles   : {unstable_tiles_all} / {total_tiles_all} ({(unstable_tiles_all/max(1, total_tiles_all))*100.0:.2f}%)")
        print(f"  Wall-clock Time  : {elapsed_time:.1f}s ({elapsed_time/60.0:.2f} min)")

    # Determine winner
    mv_pq = comparison_results["mask_vote"]["mean_pq"]
    pa_pq = comparison_results["prob_avg"]["mean_pq"]
    winner = "prob_avg" if pa_pq >= mv_pq else "mask_vote"

    print("\n============================================================")
    print(f"COMPARISON SUMMARY:")
    print(f"  mask_vote PQ : {mv_pq:.4f} (Time: {comparison_results['mask_vote']['elapsed_seconds']:.1f}s)")
    print(f"  prob_avg  PQ : {pa_pq:.4f} (Time: {comparison_results['prob_avg']['elapsed_seconds']:.1f}s)")
    print(f"  WINNER       : {winner.upper()} (PQ Delta: {pa_pq - mv_pq:+.4f})")
    print("============================================================")

    # Write Markdown Report
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w") as f:
        f.write("# Stage 1: TTA Mode Comparison Report\n\n")
        f.write(f"## Winner: **`{winner.upper()}`**\n\n")
        f.write("### Comparison Table (115 Validation Images at Threshold 0.45)\n\n")
        f.write("| Metric | `mask_vote` (Strict Majority >=3/4) | `prob_avg` (Raw Probability Averaging) | Delta (prob_avg - mask_vote) |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        f.write(f"| **Strict PQ** | **{mv_pq:.4f}** | **{pa_pq:.4f}** | **{pa_pq - mv_pq:+.4f}** |\n")
        f.write(f"| True Positives (TP) | {comparison_results['mask_vote']['tot_tp']} | {comparison_results['prob_avg']['tot_tp']} | {comparison_results['prob_avg']['tot_tp'] - comparison_results['mask_vote']['tot_tp']:+d} |\n")
        f.write(f"| False Positives (FP) | {comparison_results['mask_vote']['tot_fp']} | {comparison_results['prob_avg']['tot_fp']} | {comparison_results['prob_avg']['tot_fp'] - comparison_results['mask_vote']['tot_fp']:+d} |\n")
        f.write(f"| False Negatives (FN) | {comparison_results['mask_vote']['tot_fn']} | {comparison_results['prob_avg']['tot_fn']} | {comparison_results['prob_avg']['tot_fn'] - comparison_results['mask_vote']['tot_fn']:+d} |\n")
        f.write(f"| Mean Raw Prob Variance | {comparison_results['mask_vote']['mean_variance']:.4f} | {comparison_results['prob_avg']['mean_variance']:.4f} | {comparison_results['prob_avg']['mean_variance'] - comparison_results['mask_vote']['mean_variance']:+.4f} |\n")
        f.write(f"| Unstable Tiles (>0.05 var) | {comparison_results['mask_vote']['unstable_tiles']} | {comparison_results['prob_avg']['unstable_tiles']} | {comparison_results['prob_avg']['unstable_tiles'] - comparison_results['mask_vote']['unstable_tiles']:+d} |\n")
        f.write(f"| Total Tiles Evaluated | {comparison_results['mask_vote']['total_tiles']} | {comparison_results['prob_avg']['total_tiles']} | 0 |\n")
        f.write(f"| Wall-Clock Time | {comparison_results['mask_vote']['elapsed_seconds']:.1f}s ({comparison_results['mask_vote']['elapsed_seconds']/60.0:.1f}m) | {comparison_results['prob_avg']['elapsed_seconds']:.1f}s ({comparison_results['prob_avg']['elapsed_seconds']/60.0:.1f}m) | {comparison_results['prob_avg']['elapsed_seconds'] - comparison_results['mask_vote']['elapsed_seconds']:+.1f}s |\n\n")

        f.write("### Key Observations\n")
        f.write(f"- **Accuracy**: `prob_avg` achieved **{pa_pq:.4f}** PQ vs `mask_vote` **{mv_pq:.4f}** PQ.\n")
        f.write(f"- **Speed**: `prob_avg` completed in **{comparison_results['prob_avg']['elapsed_seconds']:.1f}s** vs **{comparison_results['mask_vote']['elapsed_seconds']:.1f}s** for `mask_vote` (post-processing executed once per tile instead of 4 times).\n")

    print(f"Report successfully saved to: {output_report}")
    return winner

if __name__ == "__main__":
    run_stage1_tta_local_comparison()
