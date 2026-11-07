import os
import json
import time
import yaml
import numpy as np
import torch
from tqdm import tqdm

from src.unet_model import UNetResNet34
from src.unet_validate import validate_unet
from src.unet_inference import predict_tiled_unet

def run_stage0_extended_sweep(
    weights_path: str = "models/unet_best.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    num_val_images: int = 20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"=== STAGE 0 — EXTENDED FREE SWEEP (FULL SPECIFICATION) ===")
    print(f"Device: {device} | Model: {weights_path}")
    print(f"============================================================")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with open(json_path, "r") as f:
        coco = json.load(f)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    val_ids = list(np.random.permutation(all_ids)[:num_val_images])

    images_dict = {img["id"]: img for img in coco["images"]}

    # Precompute probability maps for the 20 validation images once!
    print(f"Precomputing raw probability maps for the {num_val_images} validation images on GPU...")
    cached_prob_maps = {}
    for img_id in tqdm(val_ids, desc="Precomputing prob maps"):
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cached_prob_maps[img_id] = predict_tiled_unet(model, img, device)

    # -------------------------------------------------------------
    # TASK 0.1 — Extended Single-Parameter Sweeps
    # Baseline:
    #   peak_min_distance = 20
    #   prune_threshold   = 10
    #   score_threshold   = 0.50
    #   kernel_size       = 5
    #   dist_transform    = DIST_L2
    # -------------------------------------------------------------
    print("\n============================================================")
    print("TASK 0.1: EXTENDED SINGLE-PARAMETER SWEEPS")
    print("============================================================")

    results_01 = {}

    # 1. Peak min distance: [25, 30, 35, 40, 45, 50]
    print("\n--- Parameter 1/5: peak_min_distance ---")
    peak_vals = [25, 30, 35, 40, 45, 50]
    results_01["peak_min_distance"] = {}
    for val in peak_vals:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device,
            thresh=0.50, limit=num_val_images,
            postproc_kwargs={
                "peak_min_dist": val,
                "skel_prune_len": 10,
                "morph_kernel_size": 5,
                "dist_transform_type": "DIST_L2",
            },
            precomputed_prob_maps=cached_prob_maps,
        )
        results_01["peak_min_distance"][val] = res
        print(f"  peak_min_distance={val:2d} -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

    # 2. Prune threshold: [10, 15, 20, 25, 30]
    print("\n--- Parameter 2/5: prune_threshold ---")
    prune_vals = [10, 15, 20, 25, 30]
    results_01["prune_threshold"] = {}
    for val in prune_vals:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device,
            thresh=0.50, limit=num_val_images,
            postproc_kwargs={
                "peak_min_dist": 20,
                "skel_prune_len": val,
                "morph_kernel_size": 5,
                "dist_transform_type": "DIST_L2",
            },
            precomputed_prob_maps=cached_prob_maps,
        )
        results_01["prune_threshold"][val] = res
        print(f"  prune_threshold={val:2d}   -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

    # 3. Score threshold: [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    print("\n--- Parameter 3/5: score_threshold ---")
    score_vals = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results_01["score_threshold"] = {}
    for val in score_vals:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device,
            thresh=val, limit=num_val_images,
            postproc_kwargs={
                "peak_min_dist": 20,
                "skel_prune_len": 10,
                "morph_kernel_size": 5,
                "dist_transform_type": "DIST_L2",
            },
            precomputed_prob_maps=cached_prob_maps,
        )
        results_01["score_threshold"][val] = res
        print(f"  score_threshold={val:.2f} -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

    # 4. Kernel size: [3, 5, 7, 9]
    print("\n--- Parameter 4/5: kernel_size ---")
    kernel_vals = [3, 5, 7, 9]
    results_01["kernel_size"] = {}
    for val in kernel_vals:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device,
            thresh=0.50, limit=num_val_images,
            postproc_kwargs={
                "peak_min_dist": 20,
                "skel_prune_len": 10,
                "morph_kernel_size": val,
                "dist_transform_type": "DIST_L2",
            },
            precomputed_prob_maps=cached_prob_maps,
        )
        results_01["kernel_size"][val] = res
        print(f"  kernel_size={val:2d}       -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

    # 5. Distance transform: [DIST_L1, DIST_L2]
    print("\n--- Parameter 5/5: dist_transform ---")
    dt_vals = ["DIST_L1", "DIST_L2"]
    results_01["dist_transform"] = {}
    for val in dt_vals:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device,
            thresh=0.50, limit=num_val_images,
            postproc_kwargs={
                "peak_min_dist": 20,
                "skel_prune_len": 10,
                "morph_kernel_size": 5,
                "dist_transform_type": val,
            },
            precomputed_prob_maps=cached_prob_maps,
        )
        results_01["dist_transform"][val] = res
        print(f"  dist_transform={val:8s} -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

    # Best single parameters
    best_peak = max(results_01["peak_min_distance"].keys(), key=lambda k: results_01["peak_min_distance"][k]["mean_pq"])
    best_prune = max(results_01["prune_threshold"].keys(), key=lambda k: results_01["prune_threshold"][k]["mean_pq"])
    best_score = max(results_01["score_threshold"].keys(), key=lambda k: results_01["score_threshold"][k]["mean_pq"])
    best_kernel = max(results_01["kernel_size"].keys(), key=lambda k: results_01["kernel_size"][k]["mean_pq"])
    best_dt = max(results_01["dist_transform"].keys(), key=lambda k: results_01["dist_transform"][k]["mean_pq"])

    print("\n--- Task 0.1 Best Single Parameters ---")
    print(f"  Best peak_min_distance: {best_peak} (PQ: {results_01['peak_min_distance'][best_peak]['mean_pq']:.4f})")
    print(f"  Best prune_threshold:   {best_prune} (PQ: {results_01['prune_threshold'][best_prune]['mean_pq']:.4f})")
    print(f"  Best score_threshold:   {best_score} (PQ: {results_01['score_threshold'][best_score]['mean_pq']:.4f})")
    print(f"  Best kernel_size:       {best_kernel} (PQ: {results_01['kernel_size'][best_kernel]['mean_pq']:.4f})")
    print(f"  Best dist_transform:    {best_dt} (PQ: {results_01['dist_transform'][best_dt]['mean_pq']:.4f})")

    # -------------------------------------------------------------
    # TASK 0.2 — Joint Sweep
    # -------------------------------------------------------------
    print("\n============================================================")
    print("TASK 0.2: JOINT SWEEP")
    print("============================================================")

    # Set A: Peak x Prune (best ± 5)
    print("\n[Combination Set A] Peak Distance x Prune Threshold:")
    peak_candidates = sorted(list(set([max(20, best_peak - 5), best_peak, best_peak + 5])))
    prune_candidates = sorted(list(set([max(5, best_prune - 5), best_prune, best_prune + 5])))

    best_pair = (best_peak, best_prune)
    best_pair_pq = -1.0
    joint_set_a = {}

    for p in peak_candidates:
        for pr in prune_candidates:
            res = validate_unet(
                model, json_path, images_dir, val_ids, device,
                thresh=0.50, limit=num_val_images,
                postproc_kwargs={
                    "peak_min_dist": p,
                    "skel_prune_len": pr,
                    "morph_kernel_size": 5,
                    "dist_transform_type": "DIST_L2",
                },
                precomputed_prob_maps=cached_prob_maps,
            )
            joint_set_a[(p, pr)] = res
            print(f"  (Peak={p:2d}, Prune={pr:2d}) -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")
            if res["mean_pq"] > best_pair_pq:
                best_pair_pq = res["mean_pq"]
                best_pair = (p, pr)

    opt_peak, opt_prune = best_pair
    print(f"--> Best Pair (Set A): Peak={opt_peak}, Prune={opt_prune} (PQ: {best_pair_pq:.4f})")

    # Set B: (Best peak, best prune) x Score threshold [0.45, 0.50, 0.55, 0.60, 0.65]
    print("\n[Combination Set B] (Peak, Prune) x Score Threshold:")
    joint_set_b = {}
    opt_triple = (opt_peak, opt_prune, 0.50)
    best_triple_pq = -1.0
    for sc in [0.45, 0.50, 0.55, 0.60, 0.65]:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device,
            thresh=sc, limit=num_val_images,
            postproc_kwargs={
                "peak_min_dist": opt_peak,
                "skel_prune_len": opt_prune,
                "morph_kernel_size": 5,
                "dist_transform_type": "DIST_L2",
            },
            precomputed_prob_maps=cached_prob_maps,
        )
        joint_set_b[sc] = res
        print(f"  (Peak={opt_peak}, Prune={opt_prune}, Score={sc:.2f}) -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")
        if res["mean_pq"] > best_triple_pq:
            best_triple_pq = res["mean_pq"]
            opt_triple = (opt_peak, opt_prune, sc)

    opt_peak, opt_prune, opt_score = opt_triple
    print(f"--> Best Triple (Set B): Peak={opt_peak}, Prune={opt_prune}, Score={opt_score:.2f} (PQ: {best_triple_pq:.4f})")

    # Set C: (Best peak, prune, score) x Kernel size [3, 5, 7]
    print("\n[Combination Set C] (Peak, Prune, Score) x Kernel Size:")
    joint_set_c = {}
    opt_quad = (opt_peak, opt_prune, opt_score, 5)
    best_quad_pq = -1.0
    for k in [3, 5, 7]:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device,
            thresh=opt_score, limit=num_val_images,
            postproc_kwargs={
                "peak_min_dist": opt_peak,
                "skel_prune_len": opt_prune,
                "morph_kernel_size": k,
                "dist_transform_type": "DIST_L2",
            },
            precomputed_prob_maps=cached_prob_maps,
        )
        joint_set_c[k] = res
        print(f"  (Peak={opt_peak}, Prune={opt_prune}, Score={opt_score:.2f}, Kernel={k}) -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")
        if res["mean_pq"] > best_quad_pq:
            best_quad_pq = res["mean_pq"]
            opt_quad = (opt_peak, opt_prune, opt_score, k)

    opt_peak, opt_prune, opt_score, opt_kernel = opt_quad
    print(f"--> Best Quad (Set C): Peak={opt_peak}, Prune={opt_prune}, Score={opt_score:.2f}, Kernel={opt_kernel} (PQ: {best_quad_pq:.4f})")

    # Set D: (Best peak, prune, score, kernel) x Distance transform [DIST_L1, DIST_L2]
    print("\n[Combination Set D] (Peak, Prune, Score, Kernel) x Distance Transform:")
    joint_set_d = {}
    opt_quint = (opt_peak, opt_prune, opt_score, opt_kernel, "DIST_L2")
    best_quint_pq = -1.0
    for dt in ["DIST_L1", "DIST_L2"]:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device,
            thresh=opt_score, limit=num_val_images,
            postproc_kwargs={
                "peak_min_dist": opt_peak,
                "skel_prune_len": opt_prune,
                "morph_kernel_size": opt_kernel,
                "dist_transform_type": dt,
            },
            precomputed_prob_maps=cached_prob_maps,
        )
        joint_set_d[dt] = res
        print(f"  (Peak={opt_peak}, Prune={opt_prune}, Score={opt_score:.2f}, Kernel={opt_kernel}, DT={dt}) -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")
        if res["mean_pq"] > best_quint_pq:
            best_quint_pq = res["mean_pq"]
            opt_quint = (opt_peak, opt_prune, opt_score, opt_kernel, dt)

    opt_peak, opt_prune, opt_score, opt_kernel, opt_dt = opt_quint
    print(f"\n============================================================")
    print(f"FINAL OPTIMAL PRODUCTION CONFIGURATION:")
    print(f"  peak_min_distance : {opt_peak}")
    print(f"  prune_threshold   : {opt_prune}")
    print(f"  score_threshold   : {opt_score:.2f}")
    print(f"  kernel_size       : {opt_kernel}")
    print(f"  dist_transform    : {opt_dt}")
    print(f"  Final Local PQ    : {best_quint_pq:.4f}")
    print(f"============================================================")

    # -------------------------------------------------------------
    # TASK 0.3 — Lock In the Best Config
    # -------------------------------------------------------------
    config_path = "configs/unet_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["production"] = {
        "peak_min_distance": int(opt_peak),
        "prune_threshold": int(opt_prune),
        "score_threshold": float(opt_score),
        "kernel_size": int(opt_kernel),
        "dist_transform": str(opt_dt),
        "min_area": 300,
        "best_local_pq": float(best_quint_pq),
    }

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Updated {config_path} with production section.")

    # -------------------------------------------------------------
    # TASK 0.4 — Local Validation of the Production Config
    # -------------------------------------------------------------
    print("\nTASK 0.4: DETAILED LOCAL VALIDATION OF PRODUCTION CONFIG")
    prod_val_res = validate_unet(
        model, json_path, images_dir, val_ids, device,
        thresh=opt_score, limit=num_val_images,
        postproc_kwargs={
            "peak_min_dist": opt_peak,
            "skel_prune_len": opt_prune,
            "morph_kernel_size": opt_kernel,
            "dist_transform_type": opt_dt,
        },
        precomputed_prob_maps=cached_prob_maps,
    )

    print(f"Production Metrics (20 validation images):")
    print(f"  Strict PQ                   : {prod_val_res['mean_pq']:.4f}")
    print(f"  Total TP                    : {prod_val_res['total_tp']}")
    print(f"  Total FP                    : {prod_val_res['total_fp']}")
    print(f"  Total FN                    : {prod_val_res['total_fn']}")
    print(f"  Precision                   : {prod_val_res['precision']:.4f}")
    print(f"  Recall                      : {prod_val_res['recall']:.4f}")
    print(f"  Mean Instances per Image    : {prod_val_res['mean_instances_per_image']:.2f}")
    print(f"  Mean Branch Points/Comp     : {prod_val_res['mean_branch_points']:.2f}")

    # Write summary report to reports/stage0_results.md
    os.makedirs("reports", exist_ok=True)
    with open("reports/stage0_results.md", "w") as f:
        f.write("# Stage 0 — Extended Free Sweep Results\n\n")
        f.write("## 1. Optimal Production Configuration\n")
        f.write(f"- **`peak_min_distance`**: `{opt_peak}`\n")
        f.write(f"- **`prune_threshold`**: `{opt_prune}`\n")
        f.write(f"- **`score_threshold`**: `{opt_score:.2f}`\n")
        f.write(f"- **`kernel_size`**: `{opt_kernel}`\n")
        f.write(f"- **`dist_transform`**: `{opt_dt}`\n")
        f.write(f"- **Final Local Strict PQ**: **`{prod_val_res['mean_pq']:.4f}`**\n\n")

        f.write("## 2. Production Performance Metrics (20 Validation Images)\n")
        f.write(f"| Metric | Value |\n|:---|:---|\n")
        f.write(f"| Strict PQ | **{prod_val_res['mean_pq']:.4f}** |\n")
        f.write(f"| True Positives (TP) | {prod_val_res['total_tp']} |\n")
        f.write(f"| False Positives (FP) | {prod_val_res['total_fp']} |\n")
        f.write(f"| False Negatives (FN) | {prod_val_res['total_fn']} |\n")
        f.write(f"| Precision | {prod_val_res['precision']:.4f} |\n")
        f.write(f"| Recall | {prod_val_res['recall']:.4f} |\n")
        f.write(f"| Mean Instances / Image | {prod_val_res['mean_instances_per_image']:.2f} |\n")
        f.write(f"| Mean Branch Points / Comp | {prod_val_res['mean_branch_points']:.2f} |\n\n")

        f.write("## 3. Single Parameter Sweep Tables (Task 0.1)\n\n")
        for p_name, p_dict in results_01.items():
            f.write(f"### {p_name}\n")
            f.write("| Value | Strict PQ | TP | FP | FN |\n|:---|:---:|:---:|:---:|:---:|\n")
            for k_val, r in p_dict.items():
                f.write(f"| {k_val} | {r['mean_pq']:.4f} | {r['total_tp']} | {r['total_fp']} | {r['total_fn']} |\n")
            f.write("\n")

        f.write("## 4. Joint Sweeps (Task 0.2)\n")
        f.write("### Set A (Peak x Prune)\n")
        f.write("| Peak | Prune | Strict PQ | TP | FP | FN |\n|:---|:---:|:---:|:---:|:---:|\n")
        for (pk, pr), r in joint_set_a.items():
            f.write(f"| {pk} | {pr} | {r['mean_pq']:.4f} | {r['total_tp']} | {r['total_fp']} | {r['total_fn']} |\n")
        f.write("\n")

        f.write("### Set B (Score Threshold)\n")
        f.write("| Score Thresh | Strict PQ | TP | FP | FN |\n|:---|:---:|:---:|:---:|:---:|\n")
        for sc, r in joint_set_b.items():
            f.write(f"| {sc:.2f} | {r['mean_pq']:.4f} | {r['total_tp']} | {r['total_fp']} | {r['total_fn']} |\n")
        f.write("\n")

        f.write("### Set C (Kernel Size)\n")
        f.write("| Kernel Size | Strict PQ | TP | FP | FN |\n|:---|:---:|:---:|:---:|:---:|\n")
        for k, r in joint_set_c.items():
            f.write(f"| {k} | {r['mean_pq']:.4f} | {r['total_tp']} | {r['total_fp']} | {r['total_fn']} |\n")
        f.write("\n")

        f.write("### Set D (Distance Transform)\n")
        f.write("| Distance Transform | Strict PQ | TP | FP | FN |\n|:---|:---:|:---:|:---:|:---:|\n")
        for dt, r in joint_set_d.items():
            f.write(f"| {dt} | {r['mean_pq']:.4f} | {r['total_tp']} | {r['total_fp']} | {r['total_fn']} |\n")
        f.write("\n")

    print(f"Results report saved to reports/stage0_results.md")
    return cfg["production"]

if __name__ == "__main__":
    import cv2
    run_stage0_extended_sweep()
