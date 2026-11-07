import os
import sys
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from pycocotools import mask as mask_utils

# Self-contained Mask R-CNN builder
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import nms

from src.unet_postprocess import resolve_overlaps
from src.unet_validate import evaluate_strict_pq

def build_mask_rcnn(num_classes=2, box_nms_thresh=0.5):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    if box_nms_thresh is not None:
        model.roi_heads.nms_thresh = box_nms_thresh
    return model

def get_tile_coordinates(img_h=2048, img_w=2048, tile_size=512, overlap=64):
    step = tile_size - overlap
    y_starts = list(range(0, img_h - tile_size + 1, step))
    if y_starts[-1] + tile_size < img_h:
        y_starts.append(img_h - tile_size)
    x_starts = list(range(0, img_w - tile_size + 1, step))
    if x_starts[-1] + tile_size < img_w:
        x_starts.append(img_w - tile_size)
    tiles = []
    for y in y_starts:
        for x in x_starts:
            tiles.append((y, x, y + tile_size, x + tile_size))
    return tiles

def predict_tiled_maskrcnn(model, image_tensor, device, tile_size=512, overlap=64, score_thresh=0.50, mask_thresh=0.50, nms_thresh=0.50):
    _, H, W = image_tensor.shape
    tile_coords = get_tile_coordinates(H, W, tile_size, overlap)
    global_boxes, global_scores, global_masks = [], [], []

    for ymin, xmin, ymax, xmax in tile_coords:
        tile_tensor = image_tensor[:, ymin:ymax, xmin:xmax]
        with torch.no_grad():
            outputs = model([tile_tensor.to(device)])[0]

        scores = outputs["scores"].cpu().numpy()
        masks = outputs["masks"].squeeze(1).cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        boxes = outputs["boxes"].cpu().numpy()

        keep = [i for i, (s, l) in enumerate(zip(scores, labels)) if s >= score_thresh and l == 1]
        for idx in keep:
            sc = float(scores[idx])
            b = boxes[idx]
            m = (masks[idx] > mask_thresh).astype(np.uint8)

            g_box = [b[0] + xmin, b[1] + ymin, b[2] + xmin, b[3] + ymin]
            g_mask = np.zeros((H, W), dtype=np.uint8)
            g_mask[ymin:ymax, xmin:xmax] = m

            global_boxes.append(g_box)
            global_scores.append(sc)
            global_masks.append(g_mask)

    if len(global_boxes) == 0:
        return []

    t_boxes = torch.tensor(np.array(global_boxes), dtype=torch.float32)
    t_scores = torch.tensor(np.array(global_scores), dtype=torch.float32)
    keep_indices = nms(t_boxes, t_scores, iou_threshold=nms_thresh).numpy()

    final_masks = [global_masks[k] for k in keep_indices]
    if len(final_masks) > 1:
        final_masks = resolve_overlaps(final_masks, min_area=300)

    return final_masks

def audit_mask_rcnn_on_115_pool(
    weights_path: str = "models/best_model.pth",
    json_path: str = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    images_dir: str = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    score_thresh: float = 0.50,
):
    print("============================================================")
    print("=== TASK 3: AUDIT MASK R-CNN ON 115 HELD-OUT POOL ===")
    print("============================================================")

    if not os.path.exists(weights_path):
        alt_path = "archive/Mask R-CNN/models/best_model.pth"
        if os.path.exists(alt_path):
            weights_path = alt_path

    print(f"Loading Mask R-CNN from: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_mask_rcnn(num_classes=2, box_nms_thresh=0.5)
    checkpoint = torch.load(weights_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
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

    print(f"Validation pool: {len(val_ids)} images | Device: {device} | Score Thresh: {score_thresh}")

    pqs, tps, fps, fns = [], [], [], []

    for img_id in tqdm(val_ids, desc="Evaluating Mask R-CNN"):
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

        img_bgr = cv2.imread(img_p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

        pred_masks = predict_tiled_maskrcnn(
            model=model,
            image_tensor=img_tensor,
            device=device,
            tile_size=512,
            overlap=64,
            score_thresh=score_thresh,
            mask_thresh=0.50,
            nms_thresh=0.50,
        )

        res = evaluate_strict_pq(pred_masks, gt_instances)
        pqs.append(res["pq"])
        tps.append(res["tp"])
        fps.append(res["fp"])
        fns.append(res["fn"])

    mean_pq = float(np.mean(pqs))
    tot_tp = int(np.sum(tps))
    tot_fp = int(np.sum(fps))
    tot_fn = int(np.sum(fns))

    unet_pq = 0.4154
    unet_tp, unet_fp, unet_fn = 510, 346, 259

    print("\n============================================================")
    print("TASK 3 SUMMARY & COMPARISON (115 POOL):")
    print("============================================================")
    print(f"Mask R-CNN Strict PQ      : {mean_pq:.4f} (TP={tot_tp}, FP={tot_fp}, FN={tot_fn})")
    print(f"U-Net prob_avg TTA PQ     : {unet_pq:.4f} (TP={unet_tp}, FP={unet_fp}, FN={unet_fn})")
    print(f"Delta (U-Net - Mask RCNN) : {unet_pq - mean_pq:+.4f}")

    if mean_pq > 0.30:
        recommendation = "Worthwhile ensemble candidate (Mask R-CNN PQ > 0.30)."
    elif mean_pq >= 0.25:
        recommendation = "Marginal candidate (PQ 0.25 - 0.30). Test with conservative weighting in Stage 2."
    else:
        recommendation = "Likely to hurt ensemble (Mask R-CNN PQ < 0.25). Skip Stage 2 or train fresh detector."
    print(f"Decision Rule Assessment : {recommendation}")
    print("============================================================")

    return {
        "maskrcnn_pq": mean_pq,
        "tot_tp": tot_tp,
        "tot_fp": tot_fp,
        "tot_fn": tot_fn,
        "recommendation": recommendation,
    }

if __name__ == "__main__":
    audit_mask_rcnn_on_115_pool()
