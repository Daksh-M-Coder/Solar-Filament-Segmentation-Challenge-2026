import os
import json
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_inference import predict_single_image
from src.unet_validate import evaluate_strict_pq

def evaluate_prob_avg_at_threshold(
    thresh: float = 0.50,
    config_path: str = "configs/unet_config.yaml",
    weights_path: str = "models/unet_best.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
):
    print("============================================================")
    print(f"=== TASK 2: TEST PROB_AVG TTA AT THRESHOLD {thresh:.2f} (115 POOL) ===")
    print("============================================================")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    prod = cfg["production"]
    postproc_kwargs = {
        "peak_min_dist": int(prod["peak_min_distance"]),
        "skel_prune_len": int(prod["prune_threshold"]),
        "morph_kernel_size": int(prod["kernel_size"]),
        "dist_transform_type": str(prod["dist_transform"]),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    pqs, tps, fps, fns = [], [], [], []

    for img_id in tqdm(val_ids, desc=f"Evaluating prob_avg th={thresh:.2f}"):
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

        instances, _, _, _ = predict_single_image(
            model=model,
            img_path=img_p,
            device=device,
            thresh=thresh,
            use_tta=True,
            tta_mode="prob_avg",
            postproc_kwargs=postproc_kwargs,
        )

        res = evaluate_strict_pq(instances, gt_instances)
        pqs.append(res["pq"])
        tps.append(res["tp"])
        fps.append(res["fp"])
        fns.append(res["fn"])

    mean_pq = float(np.mean(pqs))
    tot_tp = int(np.sum(tps))
    tot_fp = int(np.sum(fps))
    tot_fn = int(np.sum(fns))

    # Compare with known th=0.45 result from log 9 (PQ=0.4154, TP=510, FP=346, FN=259)
    th45_pq = 0.4154
    delta = mean_pq - th45_pq

    print("\n============================================================")
    print("TASK 2 RESULTS:")
    print("============================================================")
    print(f"Local PQ at th=0.50 : {mean_pq:.4f} (TP={tot_tp}, FP={tot_fp}, FN={tot_fn})")
    print(f"Local PQ at th=0.45 : {th45_pq:.4f} (TP=510, FP=346, FN=259)")
    print(f"Delta (0.50 - 0.45) : {delta:+.4f}")
    if delta > 0:
        print("Winner: th=0.50 wins locally! (Recommend generating & uploading th=0.50)")
    else:
        print("Winner: th=0.45 remains best locally! (Keep current 0.34 submission)")
    print("============================================================")

    return {
        "thresh": thresh,
        "mean_pq": mean_pq,
        "tot_tp": tot_tp,
        "tot_fp": tot_fp,
        "tot_fn": tot_fn,
        "delta": delta,
    }

if __name__ == "__main__":
    evaluate_prob_avg_at_threshold(thresh=0.50)
