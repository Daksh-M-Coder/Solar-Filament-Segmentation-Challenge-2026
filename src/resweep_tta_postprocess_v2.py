"""
Task 3 v2: Re-Sweep Post-Processing on TTA-Averaged Probability Maps.

FIXED VERSION:
  - Uses src.unet_postprocess.postprocess_binary_mask (real skimage.skeletonize)
  - Lazy imports inside function body to avoid Win32 PyTorch multiprocessing deadlock
  - Baseline sanity check: aborts if PQ < 0.38 (means pipeline is broken)
  - flush=True on all prints for real-time log monitoring
"""
import os
import sys
import json
import numpy as np


def load_data(json_path, cache_tta_dir):
    from pycocotools import mask as mask_utils
    print("Loading COCO annotations and cached TTA probability maps...", flush=True)
    with open(json_path, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    anns_dict = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    val_ids = shuffled_ids[:max(20, int(len(all_ids) * 0.10))]
    print(f"Validation pool: {len(val_ids)} images", flush=True)

    prob_maps = {}
    gt_instances_dict = {}

    for img_id in val_ids:
        cache_p = os.path.join(cache_tta_dir, f"{img_id}.npz")
        if not os.path.exists(cache_p):
            raise FileNotFoundError(f"Missing cached TTA map: {cache_p}")
        data = np.load(cache_p)
        prob_maps[img_id] = data["prob"].astype(np.float32)

        info = images_dict[img_id]
        h, w = info["height"], info["width"]
        gts = []
        for ann in anns_dict.get(img_id, []):
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(mask_utils.merge(rles))
                gts.append(m)
            elif isinstance(seg, dict):
                m = mask_utils.decode(seg)
                gts.append(m)
        gt_instances_dict[img_id] = gts

    total_gt = sum(len(v) for v in gt_instances_dict.values())
    print(f"Loaded {len(prob_maps)} TTA maps. Total GT filaments: {total_gt}", flush=True)
    return val_ids, prob_maps, gt_instances_dict


def evaluate_config(val_ids, prob_maps, gt_instances_dict,
                    peak_min_dist, skel_prune_len, morph_kernel_size,
                    dist_transform_type, thresh, min_area=300):
    from scipy.optimize import linear_sum_assignment
    from src.unet_postprocess import postprocess_binary_mask

    def compute_iou(m1, m2):
        inter = np.logical_and(m1, m2).sum()
        if inter == 0:
            return 0.0
        return float(inter) / float(np.logical_or(m1, m2).sum())

    def strict_pq(preds, gts):
        n_p, n_g = len(preds), len(gts)
        if n_p == 0 and n_g == 0:
            return {"pq": 1.0, "tp": 0, "fp": 0, "fn": 0}
        if n_p == 0:
            return {"pq": 0.0, "tp": 0, "fp": 0, "fn": n_g}
        if n_g == 0:
            return {"pq": 0.0, "tp": 0, "fp": n_p, "fn": 0}
        iou_mat = np.zeros((n_p, n_g), dtype=np.float32)
        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                iou_mat[i, j] = compute_iou(p, g)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        tp_ious, matched_gts, matched_preds = [], set(), set()
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= 0.50:
                tp_ious.append(iou_mat[r, c])
                matched_gts.add(c)
                matched_preds.add(r)
        tp = len(matched_gts)
        fp = n_p - tp
        fn = n_g - tp
        sq = float(np.mean(tp_ious)) if tp > 0 else 0.0
        rq = float(tp) / float(tp + 0.5 * fp + 0.5 * fn)
        return {"pq": sq * rq, "tp": tp, "fp": fp, "fn": fn}

    postproc_kwargs = {
        "peak_min_dist": peak_min_dist,
        "skel_prune_len": skel_prune_len,
        "morph_kernel_size": morph_kernel_size,
        "dist_transform_type": dist_transform_type,
        "min_area": min_area,
    }

    all_pqs, tot_tp, tot_fp, tot_fn = [], 0, 0, 0
    for img_id in val_ids:
        prob = prob_maps[img_id]
        gt = gt_instances_dict[img_id]
        preds = postprocess_binary_mask(prob, thresh=thresh, **postproc_kwargs)
        res = strict_pq(preds, gt)
        all_pqs.append(res["pq"])
        tot_tp += res["tp"]
        tot_fp += res["fp"]
        tot_fn += res["fn"]

    mean_pq = float(np.mean(all_pqs))
    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    return {
        "pq": mean_pq, "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "precision": precision, "recall": recall,
        "config": {
            "peak_min_distance": peak_min_dist,
            "prune_threshold": skel_prune_len,
            "kernel_size": morph_kernel_size,
            "dist_transform": dist_transform_type,
            "score_threshold": thresh,
        },
    }


def run_postprocessing_re_sweep(
    json_path="data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    cache_tta_dir="data/cache_tta_val115",
    output_report="reports/stage1_6_task3_sweep.json",
):
    print("=================================================================", flush=True)
    print("=== TASK 3 v2: RE-SWEEP POST-PROCESSING ON TTA-AVERAGED MAPS ===", flush=True)
    print("=================================================================", flush=True)

    val_ids, prob_maps, gt_instances_dict = load_data(json_path, cache_tta_dir)
    history = []

    def evaluate_and_log(phase, peak, prune, k, dt, th, tag=""):
        res = evaluate_config(val_ids, prob_maps, gt_instances_dict,
                              peak_min_dist=peak, skel_prune_len=prune,
                              morph_kernel_size=k, dist_transform_type=dt, thresh=th)
        label = tag or f"peak={peak}, prune={prune}, k={k}, dt={dt}, th={th:.2f}"
        print(f"  {label} -> PQ: {res['pq']:.4f} | TP={res['tp']} FP={res['fp']} FN={res['fn']}", flush=True)
        history.append({"phase": phase, **res})
        return res

    # Step 0: Locked Baseline
    print("\n[Step 0] Locked Baseline (peak=55, prune=15, k=3, DIST_L1, th=0.45):", flush=True)
    baseline_res = evaluate_and_log("baseline", 55, 15, 3, "DIST_L1", 0.45, "BASELINE")
    locked_pq = baseline_res["pq"]
    print(f"  [Sanity check] Expected ~0.41, got {locked_pq:.4f}", flush=True)
    if locked_pq < 0.38:
        print("  [ABORT] Baseline PQ too low — pipeline mismatch detected!", flush=True)
        sys.exit(1)

    # Step 1: Score Threshold Sweep
    print("\n[Step 1/5] Score Threshold Sweep [0.35,0.40,0.45,0.50,0.55]:", flush=True)
    best_th, best_th_pq = 0.45, -1.0
    for th in [0.35, 0.40, 0.45, 0.50, 0.55]:
        res = evaluate_and_log("sweep_threshold", 55, 15, 3, "DIST_L1", th)
        if res["pq"] > best_th_pq:
            best_th_pq, best_th = res["pq"], th
    print(f"  >> Best threshold: th={best_th:.2f} (PQ: {best_th_pq:.4f})", flush=True)

    # Step 2: Peak Min Distance Sweep
    print(f"\n[Step 2/5] Peak Min Distance Sweep [40,45,50,55,60,65] at th={best_th:.2f}:", flush=True)
    best_peak, best_peak_pq = 55, -1.0
    for peak in [40, 45, 50, 55, 60, 65]:
        res = evaluate_and_log("sweep_peak", peak, 15, 3, "DIST_L1", best_th)
        if res["pq"] > best_peak_pq:
            best_peak_pq, best_peak = res["pq"], peak
    print(f"  >> Best peak: peak={best_peak} (PQ: {best_peak_pq:.4f})", flush=True)

    # Step 3: Prune Threshold Sweep
    print(f"\n[Step 3/5] Prune Threshold Sweep [5,10,15,20,25] at th={best_th:.2f}, peak={best_peak}:", flush=True)
    best_prune, best_prune_pq = 15, -1.0
    for prune in [5, 10, 15, 20, 25]:
        res = evaluate_and_log("sweep_prune", best_peak, prune, 3, "DIST_L1", best_th)
        if res["pq"] > best_prune_pq:
            best_prune_pq, best_prune = res["pq"], prune
    print(f"  >> Best prune: prune={best_prune} (PQ: {best_prune_pq:.4f})", flush=True)

    # Step 4: Kernel & Distance Transform Sweep
    print(f"\n[Step 4/5] Kernel & DT Sweep at th={best_th:.2f}, peak={best_peak}, prune={best_prune}:", flush=True)
    best_k, best_dt, best_morph_pq = 3, "DIST_L1", -1.0
    for dt in ["DIST_L1", "DIST_L2"]:
        for k in [3, 5, 7]:
            res = evaluate_and_log("sweep_kernel_dt", best_peak, best_prune, k, dt, best_th)
            if res["pq"] > best_morph_pq:
                best_morph_pq, best_k, best_dt = res["pq"], k, dt
    print(f"  >> Best morph: k={best_k}, dt={best_dt} (PQ: {best_morph_pq:.4f})", flush=True)

    # Step 5: Joint Fine-Tuning
    print(f"\n[Step 5/5] Joint Fine-Tuning of Top Candidates:", flush=True)
    top_peaks  = sorted(set([50, 55, 60, best_peak]))
    top_prunes = sorted(set([10, 15, best_prune]))
    top_ths    = sorted(set([0.40, 0.45, best_th]))
    top_dts    = sorted(set(["DIST_L1", best_dt]))
    top_ks     = sorted(set([3, best_k]))

    evaluated_keys = set()
    for h in history:
        c = h["config"]
        evaluated_keys.add((c["peak_min_distance"], c["prune_threshold"],
                            c["kernel_size"], c["dist_transform"], round(c["score_threshold"], 2)))

    for th in top_ths:
        for peak in top_peaks:
            for prune in top_prunes:
                for dt in top_dts:
                    for k in top_ks:
                        key = (peak, prune, k, dt, round(th, 2))
                        if key in evaluated_keys:
                            continue
                        evaluated_keys.add(key)
                        evaluate_and_log("joint_fine_tune", peak, prune, k, dt, th)

    # Summary
    sorted_history = sorted(history, key=lambda x: x["pq"], reverse=True)
    best = sorted_history[0]
    delta = best["pq"] - locked_pq

    print("\n" + "=" * 65, flush=True)
    print("TASK 3 v2 SWEEP SUMMARY & VERDICT:", flush=True)
    print("=" * 65, flush=True)
    print(f"Locked Baseline PQ   : {locked_pq:.4f}", flush=True)
    print(f"Best TTA Config PQ   : {best['pq']:.4f}", flush=True)
    print(f"Delta vs Baseline    : {delta:+.4f}", flush=True)
    print(f"Best Config          : {best['config']}", flush=True)
    print(f"Stats                : TP={best['tp']}, FP={best['fp']}, FN={best['fn']}", flush=True)
    print("-" * 65, flush=True)

    if best["pq"] > locked_pq + 0.005:
        verdict = (
            f"IMPROVEMENT FOUND (PQ={best['pq']:.4f} vs baseline {locked_pq:.4f}, "
            f"delta={delta:+.4f}). Generate new submission with best config!"
        )
    else:
        verdict = (
            f"KEEP CURRENT CONFIG (delta={delta:+.4f} < 0.005). "
            f"Current submission stands. No config change needed."
        )

    print(f"VERDICT: {verdict}", flush=True)
    print("=" * 65, flush=True)

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    output_data = {
        "baseline_pq": locked_pq,
        "best_pq": best["pq"],
        "delta": delta,
        "best_config": best["config"],
        "verdict": verdict,
        "top_10_configs": sorted_history[:10],
        "all_evaluations": history,
    }
    with open(output_report, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Full sweep log saved to: {output_report}", flush=True)
    return output_data


if __name__ == "__main__":
    run_postprocessing_re_sweep()
