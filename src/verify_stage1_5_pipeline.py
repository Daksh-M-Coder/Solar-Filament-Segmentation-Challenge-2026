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

def debug_full_115():
    print("============================================================")
    print("=== TASK A.2: RUN STAGE 1.5 PIPELINE ON 115-POOL ===")
    print("============================================================")

    config_path = "configs/unet_config.yaml"
    weights_path = "models/unet_best.pth"
    json_path = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    images_dir = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"

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
    val_ids = shuffled_ids[:max(20, int(len(all_ids) * 0.10))]

    pqs, tps, fps, fns = [], [], [], []

    for idx, img_id in enumerate(val_ids, 1):
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
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

        insts, _, _, _ = predict_single_image(
            model=model,
            img_path=img_p,
            device=device,
            thresh=thresh,
            use_tta=True,
            tta_mode="prob_avg",
            postproc_kwargs=postproc_kwargs,
        )
        res = evaluate_strict_pq(insts, gts)
        pqs.append(res["pq"])
        tps.append(res["tp"])
        fps.append(res["fp"])
        fns.append(res["fn"])
        if idx % 20 == 0 or idx == len(val_ids):
            print(f"[{idx}/{len(val_ids)}] Running mean PQ: {np.mean(pqs):.4f} (TP={np.sum(tps)}, FP={np.sum(fps)}, FN={np.sum(fns)})", flush=True)

    print("\nFINAL STAGE 1.5 VERIFICATION ON 115-POOL:")
    print(f"Mean PQ: {np.mean(pqs):.4f}")
    print(f"Total TP: {np.sum(tps)}")
    print(f"Total FP: {np.sum(fps)}")
    print(f"Total FN: {np.sum(fns)}")

if __name__ == "__main__":
    debug_full_115()
