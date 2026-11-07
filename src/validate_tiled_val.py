import argparse
import json
import os
import cv2
import numpy as np
import torch
import yaml
from pycocotools import mask as mask_utils
from skimage.measure import label

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image
from src.unet_validate import evaluate_strict_pq

def main():
    parser = argparse.ArgumentParser(description="Validate tiled U-Net on validation pool")
    parser.add_argument("--limit", type=int, default=115, help="Number of validation images")
    parser.add_argument("--score_thresh", type=float, default=None, help="Detection threshold (defaults to config production.score_threshold)")
    parser.add_argument("--min_area", type=int, default=None, help="Min area threshold (defaults to config production.min_area)")
    parser.add_argument("--use_tta", action="store_true", help="Enable TTA")
    parser.add_argument("--tta_mode", type=str, default="prob_avg", choices=["mask_vote", "prob_avg", "prob_avg_seam"])
    parser.add_argument("--checkpoint", type=str, default="models/unet_best.pth", help="Model weights path")
    parser.add_argument("--config", type=str, default="configs/unet_config.yaml", help="Path to unet_config.yaml")
    parser.add_argument("--peak_min_dist", type=int, default=None, help="Watershed peak min distance")
    parser.add_argument("--skel_prune_len", type=int, default=None, help="Skeleton prune threshold")
    parser.add_argument("--morph_kernel_size", type=int, default=None, help="Morphology kernel size")
    parser.add_argument("--dist_transform_type", type=str, default=None, help="Distance transform (DIST_L1 or DIST_L2)")
    parser.add_argument("--per_image_out", type=str, default=None, help="Path to save per-image CSV metrics")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    prod = cfg["production"]
    score_thresh = args.score_thresh if args.score_thresh is not None else float(prod.get("score_threshold", 0.45))
    min_area = args.min_area if args.min_area is not None else int(prod.get("min_area", 300))
    peak_min_dist = args.peak_min_dist if args.peak_min_dist is not None else int(prod.get("peak_min_distance", 55))
    skel_prune_len = args.skel_prune_len if args.skel_prune_len is not None else int(prod.get("prune_threshold", 15))
    morph_kernel_size = args.morph_kernel_size if args.morph_kernel_size is not None else int(prod.get("kernel_size", 3))
    dist_transform_type = args.dist_transform_type if args.dist_transform_type is not None else str(prod.get("dist_transform", "DIST_L1"))

    postproc_kwargs = {
        "peak_min_dist": peak_min_dist,
        "skel_prune_len": skel_prune_len,
        "morph_kernel_size": morph_kernel_size,
        "dist_transform_type": dist_transform_type,
        "min_area": min_area,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print(f"=== VALIDATE TILED VAL ({args.limit} IMAGES) ===")
    print("============================================================")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Device       : {device}")
    print(f"Threshold    : {score_thresh}")
    print(f"Min Area     : {min_area}")
    print(f"Use TTA      : {args.use_tta} (mode: {args.tta_mode})")
    print(f"Postproc     : {postproc_kwargs}")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

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
    val_ids = shuffled_ids[:max(20, int(len(all_ids) * 0.10))][:args.limit]

    pqs, tps, fps, fns = [], [], [], []
    fragments_per_image = []
    per_image_records = []

    for idx, img_id in enumerate(val_ids, 1):
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

        insts, _, _, _ = predict_single_image(
            model=model,
            img_path=img_p,
            device=device,
            thresh=score_thresh,
            use_tta=args.use_tta,
            tta_mode=args.tta_mode,
            postproc_kwargs=postproc_kwargs,
        )
        res = evaluate_strict_pq(insts, gt_instances)
        pqs.append(res["pq"])
        tps.append(res["tp"])
        fps.append(res["fp"])
        fns.append(res["fn"])

        # Count fragments (connected components across predicted masks)
        n_frags = 0
        for inst in insts:
            lbl, num = label(inst > 0, connectivity=2, return_num=True)
            n_frags += num
        fragments_per_image.append(n_frags)

        if args.per_image_out:
            per_image_records.append({
                "image_id": img_id,
                "file_name": info["file_name"],
                "pq": res["pq"],
                "tp": res["tp"],
                "fp": res["fp"],
                "fn": res["fn"],
                "fragments": n_frags,
            })

        if idx % 20 == 0 or idx == len(val_ids):
            print(f"[{idx}/{len(val_ids)}] Running mean PQ: {np.mean(pqs):.4f} | TP={np.sum(tps)} FP={np.sum(fps)} FN={np.sum(fns)}", flush=True)

    if args.per_image_out and per_image_records:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.per_image_out)), exist_ok=True)
        keys = ["image_id", "file_name", "pq", "tp", "fp", "fn", "fragments"]
        with open(args.per_image_out, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_image_records)
        print(f"Saved per-image metrics to: {args.per_image_out}")

    mean_pq = float(np.mean(pqs))
    tot_tp = int(np.sum(tps))
    tot_fp = int(np.sum(fps))
    tot_fn = int(np.sum(fns))
    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    mean_frags = float(np.mean(fragments_per_image)) if fragments_per_image else 0.0

    print("\n============================================================")
    print("VALIDATION SUMMARY:")
    print("============================================================")
    print(f"Strict PQ                 : {mean_pq:.4f}")
    print(f"TP                        : {tot_tp}")
    print(f"FP                        : {tot_fp}")
    print(f"FN                        : {tot_fn}")
    print(f"Precision                 : {precision:.4f}")
    print(f"Recall                    : {recall:.4f}")
    print(f"Mean Fragments per Image  : {mean_frags:.2f}")
    print("============================================================")

if __name__ == "__main__":
    main()
