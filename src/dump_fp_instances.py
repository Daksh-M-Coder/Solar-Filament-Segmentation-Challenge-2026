"""
Task C Diagnostic 1 — FP dump.
Takes unet_best.pth, runs it on the 115-pool,
collects all False Positive instances (predicted but not matched to any GT),
sorts by area (largest first), saves top-N source tiles as JPEG crops
to a folder for visual inspection.

Run:
    .\py_gpu.bat -m src.dump_fp_instances 2>&1 | Tee-Object -FilePath logs/23_fp_dump.log
"""
import os
import json
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image
from src.unet_validate import evaluate_strict_pq

# We need IoU overlap to identify which predicted instances are FP
def compute_iou(m1, m2):
    inter = np.logical_and(m1 > 0, m2 > 0).sum()
    union = np.logical_or(m1 > 0, m2 > 0).sum()
    return inter / union if union > 0 else 0.0


def match_predictions(pred_insts, gt_insts, iou_thresh=0.5):
    """Returns list of (pred_inst, is_fp, best_iou, matched_gt_idx)."""
    matched_gt = set()
    results = []
    for p in pred_insts:
        best_iou = 0.0
        best_gt = -1
        for gi, g in enumerate(gt_insts):
            iou = compute_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_thresh and best_gt not in matched_gt:
            matched_gt.add(best_gt)
            results.append((p, False, best_iou, best_gt))
        else:
            results.append((p, True, best_iou, best_gt))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/unet_best.pth")
    parser.add_argument("--limit", type=int, default=115)
    parser.add_argument("--top_n", type=int, default=20, help="Save top-N FP instances by area")
    parser.add_argument("--out_dir", type=str, default="logs/fp_dump")
    parser.add_argument("--margin", type=int, default=32, help="Padding around FP bbox when saving tile")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print("=== TASK C DIAGNOSTIC 1: FP INSTANCE DUMP ===")
    print("============================================================")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Pool       : {args.limit} images, locked cfg: peak=55, prune=15, k=3, DIST_L1, th=0.45, TTA=prob_avg")
    print(f"Output dir : {args.out_dir}")
    print(f"Top-N FPs  : {args.top_n}", flush=True)

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
    val_ids = shuffled_ids[:args.limit]

    # Collect all FP instances: (area, image_file, img_array, fp_mask)
    all_fps = []

    for idx, img_id in enumerate(val_ids, 1):
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        h, w = info["height"], info["width"]

        img_arr = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img_arr is None:
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

        pred_insts, _, _, _ = predict_single_image(
            model=model,
            img_path=img_p,
            device=device,
            thresh=0.45,
            use_tta=True,
            tta_mode="prob_avg",
            postproc_kwargs=postproc_kwargs,
        )

        if pred_insts:
            matched = match_predictions(pred_insts, gt_instances, iou_thresh=0.5)
            for (p_inst, is_fp, best_iou, _) in matched:
                if is_fp:
                    area = int(np.sum(p_inst > 0))
                    all_fps.append((area, info["file_name"], img_arr.copy(), p_inst.copy(), best_iou))

        if idx % 20 == 0 or idx == len(val_ids):
            print(f"[{idx}/{len(val_ids)}] Total FPs so far: {len(all_fps)}", flush=True)

    # Sort by area descending, take top-N
    all_fps.sort(key=lambda x: x[0], reverse=True)
    top_fps = all_fps[:args.top_n]

    print(f"\nTotal FP instances found: {len(all_fps)}")
    print(f"Saving top {len(top_fps)} by area to {args.out_dir}/")

    summary_lines = []
    for rank, (area, fname, img_arr, fp_mask, best_iou) in enumerate(top_fps, 1):
        # Bounding box of the FP instance
        rows = np.where(np.any(fp_mask > 0, axis=1))[0]
        cols = np.where(np.any(fp_mask > 0, axis=0))[0]
        r0, r1 = max(0, rows.min() - args.margin), min(img_arr.shape[0], rows.max() + args.margin)
        c0, c1 = max(0, cols.min() - args.margin), min(img_arr.shape[1], cols.max() + args.margin)

        # Crop the source image
        tile = img_arr[r0:r1, c0:c1].copy()
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

        # Overlay FP mask in red
        fp_crop = fp_mask[r0:r1, c0:c1]
        tile_rgb[fp_crop > 0] = (0, 0, 220)  # red

        out_name = os.path.join(args.out_dir, f"fp_{rank:02d}_area{area}_iou{best_iou:.2f}_{os.path.splitext(fname)[0]}.jpg")
        cv2.imwrite(out_name, tile_rgb)
        summary_lines.append(f"Rank {rank:2d} | area={area:6d} | best_iou={best_iou:.3f} | file={fname}")

    print("\n=== FP DUMP SUMMARY (top instances) ===")
    for line in summary_lines:
        print(line)
    print(f"\nFP crop images saved to: {args.out_dir}/")

    # Save summary to txt
    with open(os.path.join(args.out_dir, "fp_summary.txt"), "w") as f:
        f.write(f"Total FPs found across {args.limit} images: {len(all_fps)}\n")
        f.write(f"Top-{args.top_n} saved:\n\n")
        for line in summary_lines:
            f.write(line + "\n")
    print("Summary saved to fp_dump/fp_summary.txt")


if __name__ == "__main__":
    main()
