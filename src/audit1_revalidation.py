import os
import json
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_tiled_unet
from src.unet_postprocess import postprocess_binary_mask
from src.unet_validate import evaluate_strict_pq

def run_audit1_revalidation(
    config_path: str = "configs/unet_config.yaml",
    weights_path: str = "models/unet_best.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    output_report: str = "reports/stage0_5_audit1_revalidation.md",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    prod = cfg["production"]
    score_thresh = float(prod["score_threshold"])      # 0.45
    peak_min_dist = int(prod["peak_min_distance"])      # 55
    prune_len = int(prod["prune_threshold"])            # 15
    kernel_size = int(prod["kernel_size"])              # 3
    dist_tf = str(prod["dist_transform"])               # DIST_L1

    print("============================================================")
    print("=== AUDIT 1: RE-VALIDATION ON FULL 115 HELD-OUT POOL ===")
    print("============================================================")
    print(f"Locked Production Config: peak={peak_min_dist}, prune={prune_len}, score={score_thresh}, kernel={kernel_size}, dt={dist_tf}")

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
    n_val = max(20, int(len(all_ids) * 0.10))  # Exactly 115 images
    val_ids = shuffled_ids[:n_val]
    sweep_20_ids = set(shuffled_ids[:20])
    unseen_95_ids = set(shuffled_ids[20:n_val])

    print(f"Total Validation Images: {len(val_ids)} (Original 20-image subset: {len(sweep_20_ids)}, Unseen held-out: {len(unseen_95_ids)})")

    postproc_kwargs = {
        "peak_min_dist": peak_min_dist,
        "skel_prune_len": prune_len,
        "morph_kernel_size": kernel_size,
        "dist_transform_type": dist_tf,
    }

    per_image_results = []
    pqs_all, tps_all, fps_all, fns_all = [], [], [], []
    pqs_20, tps_20, fps_20, fns_20 = [], [], [], []
    pqs_95, tps_95, fps_95, fns_95 = [], [], [], []
    inst_counts = []
    branch_counts = []

    for idx, img_id in enumerate(tqdm(val_ids, desc="Auditing 115 val images"), 1):
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        h, w = info["height"], info["width"]

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

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

        prob_map = predict_tiled_unet(model, img, device)
        pred_instances = postprocess_binary_mask(prob_map, thresh=score_thresh, **postproc_kwargs)

        res = evaluate_strict_pq(pred_instances, gt_instances)
        pq, tp, fp, fn = res["pq"], res["tp"], res["fp"], res["fn"]

        pqs_all.append(pq)
        tps_all.append(tp)
        fps_all.append(fp)
        fns_all.append(fn)
        inst_counts.append(len(pred_instances))

        if img_id in sweep_20_ids:
            pqs_20.append(pq)
            tps_20.append(tp)
            fps_20.append(fp)
            fns_20.append(fn)
        else:
            pqs_95.append(pq)
            tps_95.append(tp)
            fps_95.append(fp)
            fns_95.append(fn)

        per_image_results.append({
            "img_id": img_id,
            "file_name": info["file_name"],
            "pq": pq,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_preds": len(pred_instances),
            "n_gts": len(gt_instances),
            "is_sweep_subset": img_id in sweep_20_ids,
        })

    mean_pq_all = float(np.mean(pqs_all))
    tot_tp_all = int(np.sum(tps_all))
    tot_fp_all = int(np.sum(fps_all))
    tot_fn_all = int(np.sum(fns_all))
    prec_all = tot_tp_all / (tot_tp_all + tot_fp_all) if (tot_tp_all + tot_fp_all) > 0 else 0.0
    rec_all = tot_tp_all / (tot_tp_all + tot_fn_all) if (tot_tp_all + tot_fn_all) > 0 else 0.0

    mean_pq_20 = float(np.mean(pqs_20))
    tot_tp_20 = int(np.sum(tps_20))
    tot_fp_20 = int(np.sum(fps_20))
    tot_fn_20 = int(np.sum(fns_20))

    mean_pq_95 = float(np.mean(pqs_95))
    tot_tp_95 = int(np.sum(tps_95))
    tot_fp_95 = int(np.sum(fps_95))
    tot_fn_95 = int(np.sum(fns_95))

    p25, p50, p75 = np.percentile(pqs_all, [25, 50, 75])

    print("\n============================================================")
    print(f"AUDIT 1 SUMMARY RESULTS:")
    print(f"  Full 115 Images Strict PQ      : {mean_pq_all:.4f} (TP={tot_tp_all}, FP={tot_fp_all}, FN={tot_fn_all})")
    print(f"  Original 20 Images Strict PQ   : {mean_pq_20:.4f} (TP={tot_tp_20}, FP={tot_fp_20}, FN={tot_fn_20})")
    print(f"  Unseen 95 Images Strict PQ     : {mean_pq_95:.4f} (TP={tot_tp_95}, FP={tot_fp_95}, FN={tot_fn_95})")
    print(f"  Precision: {prec_all:.4f} | Recall: {rec_all:.4f}")
    print(f"  PQ Percentiles: 25th={p25:.4f}, Median={p50:.4f}, 75th={p75:.4f}")
    print(f"  Mean Instances per Image: {np.mean(inst_counts):.2f}")
    print("============================================================")

    # Decision rule check
    if mean_pq_all >= 0.40:
        case = "Case A (PQ >= 0.40: Config is real and generalizable! Proceed to Audit 2 with confidence)"
    elif mean_pq_all >= 0.35:
        case = "Case B (PQ 0.35 - 0.40: Partial overfitting; still strong net improvement. Proceed to Audit 2)"
    else:
        case = "Case C (PQ < 0.35: Significant overfitting; STOP and report to human)"

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w") as f:
        f.write("# Audit 1: Re-Validation on Full Held-Out Pool (115 Images)\n\n")
        f.write(f"## Decision: **{case}**\n\n")
        f.write("### Locked Production Configuration Evaluated\n")
        f.write(f"- `peak_min_distance`: {peak_min_dist}\n")
        f.write(f"- `prune_threshold`: {prune_len}\n")
        f.write(f"- `score_threshold`: {score_thresh}\n")
        f.write(f"- `kernel_size`: {kernel_size}\n")
        f.write(f"- `dist_transform`: {dist_tf}\n\n")

        f.write("### Comparative Evaluation Table\n")
        f.write("| Split Pool | Images | Strict PQ | TP | FP | FN | Precision | Recall |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **Full Held-Out Pool** | **115** | **{mean_pq_all:.4f}** | **{tot_tp_all}** | **{tot_fp_all}** | **{tot_fn_all}** | **{prec_all:.4f}** | **{rec_all:.4f}** |\n")
        f.write(f"| Original Sweep Subset | 20 | {mean_pq_20:.4f} | {tot_tp_20} | {tot_fp_20} | {tot_fn_20} | {tot_tp_20/(tot_tp_20+tot_fp_20):.4f} | {tot_tp_20/(tot_tp_20+tot_fn_20):.4f} |\n")
        f.write(f"| Unseen Held-Out Subset | 95 | {mean_pq_95:.4f} | {tot_tp_95} | {tot_fp_95} | {tot_fn_95} | {tot_tp_95/(tot_tp_95+tot_fp_95):.4f} | {tot_tp_95/(tot_tp_95+tot_fn_95):.4f} |\n\n")

        f.write("### Distribution of Per-Image Strict PQ (115 Images)\n")
        f.write(f"- **Min**: {np.min(pqs_all):.4f}\n")
        f.write(f"- **25th Percentile**: {p25:.4f}\n")
        f.write(f"- **Median (50th Percentile)**: {p50:.4f}\n")
        f.write(f"- **75th Percentile**: {p75:.4f}\n")
        f.write(f"- **Max**: {np.max(pqs_all):.4f}\n")
        f.write(f"- **Mean Instances / Image**: {np.mean(inst_counts):.2f}\n\n")

        f.write("### Per-Image Detailed Breakdown\n")
        f.write("| Image Filename | Pool | Strict PQ | TP | FP | FN | Preds | GTs |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in per_image_results:
            pool_tag = "Sweep-20" if r["is_sweep_subset"] else "Unseen-95"
            f.write(f"| {r['file_name']} | {pool_tag} | {r['pq']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['n_preds']} | {r['n_gts']} |\n")

    print(f"Audit 1 report written to: {output_report}")
    return mean_pq_all

if __name__ == "__main__":
    run_audit1_revalidation()
