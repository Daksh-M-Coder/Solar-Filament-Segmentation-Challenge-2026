import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_utils

class SolarFilamentUNetDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        images_dir: str,
        image_ids: list = None,
        crop_size: int = 512,
        is_train: bool = True,
        spine_dilation: int = 3,
    ):
        self.images_dir = images_dir
        self.crop_size = crop_size
        self.is_train = is_train
        self.spine_dilation = spine_dilation

        with open(json_path, "r") as f:
            coco = json.load(f)

        self.images_dict = {img["id"]: img for img in coco["images"]}
        
        # Group annotations by image_id
        self.anns_dict = {}
        for ann in coco["annotations"]:
            self.anns_dict.setdefault(ann["image_id"], []).append(ann)

        all_ids = list(self.images_dict.keys())
        if image_ids is not None:
            self.image_ids = [i for i in image_ids if i in self.images_dict]
        else:
            self.image_ids = all_ids

        # Mean and std for grayscale ImageNet normalization
        self.mean = 0.449
        self.std = 0.226

    def __len__(self):
        # 1000 steps per epoch during training
        return len(self.image_ids) * 2 if self.is_train else len(self.image_ids)

    def _rasterize_annotations(self, anns, height, width):
        main_mask = np.zeros((height, width), dtype=np.uint8)
        spine_mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                # Polygons
                rles = mask_utils.frPyObjects(seg, height, width)
                rle = mask_utils.merge(rles)
                m = mask_utils.decode(rle)
                main_mask = np.maximum(main_mask, m)
            elif isinstance(seg, dict):
                m = mask_utils.decode(seg)
                main_mask = np.maximum(main_mask, m)

            # Spine polyline
            spine_pts = ann.get("spine", [])
            if spine_pts and len(spine_pts) >= 4:
                pts = np.array(spine_pts, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.polylines(spine_mask, [pts], isClosed=False, color=1, thickness=1)

        # Dilate spine mask to 3 pixels so it survives augmentation safely
        if self.spine_dilation > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.spine_dilation, self.spine_dilation))
            spine_mask = cv2.dilate(spine_mask, kernel, iterations=1)

        return main_mask, spine_mask

    def __getitem__(self, idx):
        # Select image
        img_id = self.image_ids[idx % len(self.image_ids)]
        img_info = self.images_dict[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback black canvas
            img = np.zeros((2048, 2048), dtype=np.uint8)

        h, w = img.shape
        anns = self.anns_dict.get(img_id, [])
        main_mask, spine_mask = self._rasterize_annotations(anns, h, w)

        if not self.is_train:
            # For validation return full images or center crop
            norm_img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
            return (
                torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0),
                torch.tensor(main_mask, dtype=torch.float32).unsqueeze(0),
                torch.tensor(spine_mask, dtype=torch.float32).unsqueeze(0),
                img_id,
            )

        # 50% positive crops (centered on a filament), 50% random crops
        cs = self.crop_size
        is_positive_sampling = (random.random() < 0.5) and (len(anns) > 0)

        if is_positive_sampling:
            ann = random.choice(anns)
            bx, by, bw, bh = ann["bbox"]
            cx = int(bx + bw / 2.0)
            cy = int(by + bh / 2.0)
            
            # Add small random jitter around centroid
            jitter_x = random.randint(-cs // 4, cs // 4)
            jitter_y = random.randint(-cs // 4, cs // 4)
            x1 = max(0, min(w - cs, cx - cs // 2 + jitter_x))
            y1 = max(0, min(h - cs, cy - cs // 2 + jitter_y))
        else:
            x1 = random.randint(0, max(0, w - cs))
            y1 = random.randint(0, max(0, h - cs))

        x2, y2 = x1 + cs, y1 + cs
        crop_img = img[y1:y2, x1:x2].copy()
        crop_main = main_mask[y1:y2, x1:x2].copy()
        crop_spine = spine_mask[y1:y2, x1:x2].copy()

        # Augmentations
        # 1. H-flip
        if random.random() < 0.5:
            crop_img = np.fliplr(crop_img).copy()
            crop_main = np.fliplr(crop_main).copy()
            crop_spine = np.fliplr(crop_spine).copy()

        # 2. V-flip
        if random.random() < 0.5:
            crop_img = np.flipud(crop_img).copy()
            crop_main = np.flipud(crop_main).copy()
            crop_spine = np.flipud(crop_spine).copy()

        # 3. 90-deg rotations
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            crop_img = np.rot90(crop_img, k=rot_k).copy()
            crop_main = np.rot90(crop_main, k=rot_k).copy()
            crop_spine = np.rot90(crop_spine, k=rot_k).copy()

        # 4. Brightness jitter [0.9, 1.1] (image only)
        scale_bright = random.uniform(0.9, 1.1)
        crop_img = np.clip(crop_img.astype(np.float32) * scale_bright, 0, 255).astype(np.uint8)

        # 5. Scale jitter [0.9, 1.1] with NEAREST for masks
        scale_j = random.uniform(0.9, 1.1)
        new_w = int(cs * scale_j)
        new_h = int(cs * scale_j)
        res_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        res_main = cv2.resize(crop_main, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        res_spine = cv2.resize(crop_spine, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad or crop back to cs x cs
        final_img = np.zeros((cs, cs), dtype=np.uint8)
        final_main = np.zeros((cs, cs), dtype=np.uint8)
        final_spine = np.zeros((cs, cs), dtype=np.uint8)

        min_w, min_h = min(cs, new_w), min(cs, new_h)
        final_img[:min_h, :min_w] = res_img[:min_h, :min_w]
        final_main[:min_h, :min_w] = res_main[:min_h, :min_w]
        final_spine[:min_h, :min_w] = res_spine[:min_h, :min_w]

        # Normalization
        norm_img = (final_img.astype(np.float32) / 255.0 - self.mean) / self.std

        return (
            torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(final_main, dtype=torch.float32).unsqueeze(0),
            torch.tensor(final_spine, dtype=torch.float32).unsqueeze(0),
        )

if __name__ == "__main__":
    json_p = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    imgs_dir = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
    ds = SolarFilamentUNetDataset(json_p, imgs_dir, crop_size=512, is_train=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    img_b, main_b, spine_b = next(iter(loader))
    print(f"Dataset Verification Successful!")
    print(f"Batch img shape: {img_b.shape}, mean: {img_b.mean():.4f}, std: {img_b.std():.4f}")
    print(f"Batch main mask shape: {main_b.shape}, fg pixels: {(main_b > 0).sum().item()}")
    print(f"Batch spine mask shape: {spine_b.shape}, spine pixels: {(spine_b > 0).sum().item()}")
