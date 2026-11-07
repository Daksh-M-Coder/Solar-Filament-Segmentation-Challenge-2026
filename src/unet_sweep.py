import os
import json
import numpy as np
import torch

from src.unet_model import UNetResNet34
from src.unet_validate import validate_unet

def run_task8_hyperparam_sweep(
    weights_path: str = "models/unet_best.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    num_val_images: int = 20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TASK 8: HYPERPARAMETER VALIDATION SWEEP (GPU: {device}) ===")

    model = UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with open(json_path, "r") as f:
        coco = json.load(f)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    val_ids = list(np.random.permutation(all_ids)[:num_val_images])

    # 1. Sweep Score Thresholds [0.40, 0.45, 0.50, 0.55, 0.60]
    print("\n[Sweep 1/5] Score Thresholds:")
    for th in [0.40, 0.45, 0.50, 0.55, 0.60]:
        res = validate_unet(model, json_path, images_dir, val_ids, device, thresh=th, limit=num_val_images)
        print(f"  Thresh {th:.2f} -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

    # 2. Sweep Peak Min Distance [15, 20, 25, 30]
    print("\n[Sweep 2/5] Peak Min Distance:")
    for pdist in [15, 20, 25, 30]:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device, thresh=0.50, limit=num_val_images,
            postproc_kwargs={"peak_min_dist": pdist}
        )
        print(f"  Peak Dist {pdist} -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

    # 3. Sweep Morphological Kernel Size [3, 5, 7]
    print("\n[Sweep 3/5] Morphological Kernel Size:")
    for k in [3, 5, 7]:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device, thresh=0.50, limit=num_val_images,
            postproc_kwargs={"morph_kernel_size": k}
        )
        print(f"  Kernel {k} -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

    # 4. Sweep Distance Transform [DIST_L1, DIST_L2]
    print("\n[Sweep 4/5] Distance Transform:")
    for dt in ["DIST_L1", "DIST_L2"]:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device, thresh=0.50, limit=num_val_images,
            postproc_kwargs={"dist_transform_type": dt}
        )
        print(f"  Transform {dt} -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

    # 5. Sweep Skeleton Prune Threshold [5, 10, 15]
    print("\n[Sweep 5/5] Skeleton Prune Threshold:")
    for prune in [5, 10, 15]:
        res = validate_unet(
            model, json_path, images_dir, val_ids, device, thresh=0.50, limit=num_val_images,
            postproc_kwargs={"skel_prune_len": prune}
        )
        print(f"  Prune {prune} -> PQ: {res['mean_pq']:.4f} | TP: {res['total_tp']} | FP: {res['total_fp']} | FN: {res['total_fn']}")

if __name__ == "__main__":
    run_task8_hyperparam_sweep()
