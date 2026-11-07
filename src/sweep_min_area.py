"""
Task A — min_area sweep on 115-pool.
Sweeps min_area over {99999, 100, 200, 300, 500, 800} using the locked config:
  peak_min_dist=55, skel_prune_len=15, morph_kernel_size=3, DIST_L1, thresh=0.45, TTA prob_avg.

Run:
    .\py_gpu.bat -m src.sweep_min_area 2>&1 | Tee-Object -FilePath logs/22_min_area_sweep.log
"""
import os
import json
import argparse
import csv
import numpy as np
import torch
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image
from src.unet_validate import evaluate_strict_pq
from skimage.measure import label


def load_model(checkpoint_path: str, device: torch.device):
    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def run_single_pass(model, val_ids, images_dict, anns_dict, images_dir, device, min_area):
    """Run full 115-pool validation with a given min_area, locked config."""
    postproc_kwargs = {
        "peak_min_dist": 55,
        "skel_prune_len": 15,
        "morph_kernel_size": 3,
        "dist_transform_type": "DIST_L1",
        "min_area": min_area,
    }
    pqs, tps, fps, fns, frags = [], [], [], [], []
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
            thresh=0.45,
            use_tta=True,
            tta_mode="prob_avg",
            postproc_kwargs=postproc_kwargs,
        )
        res = evaluate_strict_pq(insts, gt_instances)
        pqs.append(res["pq"])
        tps.append(res["tp"])
        fps.append(res["fp"])
        fns.append(res["fn"])

        n_frags = 0
        for inst in insts:
            _, num = label(inst > 0, connectivity=2, return_num=True)
            n_frags += num
        frags.append(n_frags)

        if idx % 20 == 0 or idx == len(val_ids):
            print(f"  [{idx}/{len(val_ids)}] Running PQ: {np.mean(pqs):.4f}", flush=True)

    return {
        "mean_pq": float(np.mean(pqs)),
        "tp": int(np.sum(tps)),
        "fp": int(np.sum(fps)),
        "fn": int(np.sum(fns)),
        "mean_frags": float(np.mean(frags)),
    }


def main():
    parser = argparse.ArgumentParser(description="min_area sweep on 115-pool")
    parser.add_argument("--checkpoint", type=str, default="models/unet_best.pth")
    parser.add_argument("--limit", type=int, default=115)
    parser.add_argument("--skip_sanity", action="store_true", help="Skip min_area=99999 sanity check")
    parser.add_argument("--out_csv", type=str, default="logs/22_min_area_sweep_results.csv")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print("=== TASK A: min_area SWEEP (115-POOL, LIVE TTA) ===")
    print("============================================================")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Device      : {device}")
    print(f"Locked cfg  : peak=55, prune=15, k=3, DIST_L1, th=0.45, TTA=prob_avg")
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
    val_ids = shuffled_ids[:args.limit]

    model = load_model(args.checkpoint, device)

    results = []

    # --- Sanity check: min_area = 99999 must collapse PQ to 0 ---
    if not args.skip_sanity:
        print("\n[SANITY CHECK] Running with min_area=99999 -- PQ should collapse to 0 or near 0...", flush=True)
        r = run_single_pass(model, val_ids, images_dict, anns_dict, images_dir, device, min_area=99999)
        print(f"  SANITY: min_area=99999 -> PQ={r['mean_pq']:.4f} TP={r['tp']} FP={r['fp']} FN={r['fn']}")
        if r["mean_pq"] > 0.05:
            print("  *** WARNING: min_area wiring may be broken -- PQ did not collapse with min_area=99999 ***")
        else:
            print("  SANITY PASSED: wiring confirmed.")
        results.append({"min_area": 99999, **r})

    # --- Main sweep ---
    sweep_values = [100, 200, 300, 500, 800]
    for min_area in sweep_values:
        print(f"\n--- min_area = {min_area} ---", flush=True)
        r = run_single_pass(model, val_ids, images_dict, anns_dict, images_dir, device, min_area=min_area)
        print(f"  min_area={min_area} -> PQ={r['mean_pq']:.4f} | TP={r['tp']} FP={r['fp']} FN={r['fn']} | Frags={r['mean_frags']:.2f}", flush=True)
        results.append({"min_area": min_area, **r})

    # --- Print results table ---
    print("\n============================================================")
    print("min_area SWEEP RESULTS TABLE:")
    print("============================================================")
    print(f"{'min_area':>10} | {'PQ':>8} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Frags':>7}")
    print("-" * 58)
    for r in results:
        flag = " *** BEST ***" if r["mean_pq"] > 0.4154 else ""
        print(f"{r['min_area']:>10} | {r['mean_pq']:>8.4f} | {r['tp']:>6} | {r['fp']:>6} | {r['fn']:>6} | {r['mean_frags']:>7.2f}{flag}")
    print("============================================================")
    print("Baseline (locked): min_area=300 -> PQ=0.4154 | TP=510 FP=346 FN=259 | Frags=7.44")

    # Decision
    real_results = [r for r in results if r["min_area"] != 99999]
    best = max(real_results, key=lambda x: x["mean_pq"])
    if best["mean_pq"] > 0.4154 + 0.005:
        print(f"\nDECISION: ADOPT min_area={best['min_area']} (PQ={best['mean_pq']:.4f}, delta=+{best['mean_pq']-0.4154:.4f})")
    else:
        print(f"\nDECISION: KEEP locked min_area=300 (no candidate beat baseline by >0.005)")

    # Save CSV
    keys = ["min_area", "mean_pq", "tp", "fp", "fn", "mean_frags"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
