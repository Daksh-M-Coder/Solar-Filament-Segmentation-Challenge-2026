import os
import json
import cv2
import numpy as np
import torch
import yaml
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image
from src.unet_validate import evaluate_strict_pq
from src.unet_postprocess import postprocess_binary_mask

def debug_discrepancy():
    print("============================================================")
    print("=== TASK A.2: INVESTIGATE BASELINE DISCREPANCY (5 IMAGES) ===")
    print("============================================================")

    config_path = "configs/unet_config.yaml"
    weights_path = "models/unet_best.pth"
    json_path = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    images_dir = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
    cache_tta_dir = "data/cache_tta_val115"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    prod = cfg["production"]
    peak_min_dist = int(prod["peak_min_distance"])  # 55
    prune_len = int(prod["prune_threshold"])        # 15
    kernel_size = int(prod["kernel_size"])          # 3
    dist_tf = str(prod["dist_transform"])           # DIST_L1
    thresh = 0.45

    postproc_kwargs = {
        "peak_min_dist": peak_min_dist,
        "skel_prune_len": prune_len,
        "morph_kernel_size": kernel_size,
        "dist_transform_type": dist_tf,
    }
    print(f"Config: peak={peak_min_dist}, prune={prune_len}, k={kernel_size}, dt={dist_tf}, th={thresh}")

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
    val_ids = shuffled_ids[:max(20, int(len(all_ids) * 0.10))]
    sample_ids = val_ids[:5]
    print(f"Testing sample 5 image IDs: {sample_ids}")

    s15_pqs = []
    s16_pqs = []
    hybrid_pqs = []

    for img_id in sample_ids:
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        h, w = info["height"], info["width"]

        # GT instances
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

        # 1. Stage 1.5 Pipeline: predict_single_image(use_tta=True, tta_mode="prob_avg")
        s15_insts, _, _, _ = predict_single_image(
            model=model,
            img_path=img_p,
            device=device,
            thresh=thresh,
            use_tta=True,
            tta_mode="prob_avg",
            postproc_kwargs=postproc_kwargs,
        )
        res_15 = evaluate_strict_pq(s15_insts, gts)
        s15_pqs.append(res_15["pq"])

        # 2. Stage 1.6 Pipeline: postprocess_binary_mask on cached TTA map
        cache_p = os.path.join(cache_tta_dir, f"{img_id}.npz")
        data = np.load(cache_p)
        cached_prob = data["prob"].astype(np.float32)
        s16_insts = postprocess_binary_mask(cached_prob, thresh=thresh, **postproc_kwargs)
        res_16 = evaluate_strict_pq(s16_insts, gts)
        s16_pqs.append(res_16["pq"])

        # 3. Hybrid: What if we recompute TTA prob map directly and run postprocess_binary_mask?
        # Let's inspect max diff between cached_prob and on-the-fly TTA prob
        from src.cache_tta_val115 import compute_image_tta_prob_map
        img_gray = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        live_prob = compute_image_tta_prob_map(model, img_gray, device)
        max_prob_diff = float(np.max(np.abs(cached_prob - live_prob)))

        hybrid_insts = postprocess_binary_mask(live_prob, thresh=thresh, **postproc_kwargs)
        res_hybrid = evaluate_strict_pq(hybrid_insts, gts)
        hybrid_pqs.append(res_hybrid["pq"])

        print(f"\nImage ID {img_id}:")
        print(f"  Stage 1.5 (predict_single_image) : PQ={res_15['pq']:.4f} (TP={res_15['tp']}, FP={res_15['fp']}, FN={res_15['fn']}, Preds={len(s15_insts)})")
        print(f"  Stage 1.6 (cached map + postproc): PQ={res_16['pq']:.4f} (TP={res_16['tp']}, FP={res_16['fp']}, FN={res_16['fn']}, Preds={len(s16_insts)})")
        print(f"  Live TTA map + postproc          : PQ={res_hybrid['pq']:.4f} (TP={res_hybrid['tp']}, FP={res_hybrid['fp']}, FN={res_hybrid['fn']}, Preds={len(hybrid_insts)})")
        print(f"  Max diff (cached vs live prob)   : {max_prob_diff:.6f}")

    print("\n============================================================")
    print("5-IMAGE SUMMARY:")
    print(f"Mean Stage 1.5 PQ : {np.mean(s15_pqs):.4f}")
    print(f"Mean Stage 1.6 PQ : {np.mean(s16_pqs):.4f}")
    print(f"Mean Hybrid PQ    : {np.mean(hybrid_pqs):.4f}")
    print("============================================================")

if __name__ == "__main__":
    debug_discrepancy()
