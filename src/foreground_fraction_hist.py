"""
Task C Diagnostic 2 — Foreground-fraction histogram.
Counts what fraction of training crops contain foreground pixels,
broken into buckets: 0%, 0-1%, 1-5%, 5-20%, >20%.
Runs on 2000 sampled crops (deterministic with seed 0).

Run:
    .\py_gpu.bat -m src.foreground_fraction_hist 2>&1 | Tee-Object -FilePath logs/24_fg_fraction_hist.log
"""
import os
import json
import random
import numpy as np
import cv2
from pycocotools import mask as mask_utils


def sample_crops(json_path, images_dir, image_ids, n_samples=2000, crop_size=512, seed=0):
    """
    Samples n_samples crops from training images using the EXACT same
    50/50 positive/random crop strategy as SolarFilamentUNetDataset.
    Returns list of foreground_fraction (float) for each crop.
    """
    random.seed(seed)
    np.random.seed(seed)

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    anns_dict = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    fg_fracs = []
    cs = crop_size

    ids = list(image_ids)
    for i in range(n_samples):
        img_id = ids[i % len(ids)]
        img_info = images_dict[img_id]
        img_path = os.path.join(images_dir, img_info["file_name"])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            fg_fracs.append(0.0)
            continue

        h, w = img.shape
        anns = anns_dict.get(img_id, [])

        # Build main mask
        main_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(mask_utils.merge(rles))
                main_mask = np.maximum(main_mask, m)
            elif isinstance(seg, dict):
                m = mask_utils.decode(seg)
                main_mask = np.maximum(main_mask, m)

        # Crop strategy: 50% positive, 50% random
        is_positive_sampling = (random.random() < 0.5) and (len(anns) > 0)
        if is_positive_sampling:
            ann = random.choice(anns)
            bx, by, bw, bh = ann["bbox"]
            cx = int(bx + bw / 2.0)
            cy = int(by + bh / 2.0)
            jitter_x = random.randint(-cs // 4, cs // 4)
            jitter_y = random.randint(-cs // 4, cs // 4)
            x1 = max(0, min(w - cs, cx - cs // 2 + jitter_x))
            y1 = max(0, min(h - cs, cy - cs // 2 + jitter_y))
        else:
            x1 = random.randint(0, max(0, w - cs))
            y1 = random.randint(0, max(0, h - cs))

        x2, y2 = x1 + cs, y1 + cs
        crop_main = main_mask[y1:y2, x1:x2]

        fg_pixels = int(np.sum(crop_main > 0))
        total_pixels = crop_main.size
        fg_frac = fg_pixels / total_pixels
        fg_fracs.append(fg_frac)

    return fg_fracs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("============================================================")
    print("=== TASK C DIAGNOSTIC 2: FOREGROUND FRACTION HISTOGRAM ===")
    print("============================================================")
    print(f"Samples : {args.n_samples} crops (50/50 positive/random, same as training)")
    print(f"Seed    : {args.seed}", flush=True)

    json_path = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    images_dir = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"

    with open(json_path, "r") as f:
        coco = json.load(f)

    all_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = list(np.random.permutation(all_ids))
    n_val = max(20, int(len(all_ids) * 0.10))
    train_ids = shuffled_ids[n_val:]  # same split as unet_train.py / train_cldice.py

    print(f"Training pool: {len(train_ids)} images", flush=True)

    fg_fracs = sample_crops(
        json_path=json_path,
        images_dir=images_dir,
        image_ids=train_ids,
        n_samples=args.n_samples,
        crop_size=args.crop_size,
        seed=args.seed,
    )

    fg_arr = np.array(fg_fracs)

    # Buckets
    n_zero  = int(np.sum(fg_arr == 0.0))
    n_tiny  = int(np.sum((fg_arr > 0.0) & (fg_arr <= 0.01)))
    n_low   = int(np.sum((fg_arr > 0.01) & (fg_arr <= 0.05)))
    n_med   = int(np.sum((fg_arr > 0.05) & (fg_arr <= 0.20)))
    n_high  = int(np.sum(fg_arr > 0.20))
    total   = len(fg_arr)

    print("\n============================================================")
    print("FOREGROUND FRACTION HISTOGRAM:")
    print("============================================================")
    print(f"{'Bucket':>20} | {'Count':>7} | {'%':>6}")
    print("-" * 40)
    for label, count in [
        ("0%  (empty)", n_zero),
        ("0-1%  (trace)", n_tiny),
        ("1-5%  (sparse)", n_low),
        ("5-20%  (moderate)", n_med),
        (">20%  (dense)", n_high),
    ]:
        print(f"{label:>20} | {count:>7} | {100.0 * count / total:>5.1f}%")
    print("============================================================")
    print(f"Mean FG fraction   : {fg_arr.mean()*100:.2f}%")
    print(f"Median FG fraction : {np.median(fg_arr)*100:.2f}%")
    print(f"Std FG fraction    : {fg_arr.std()*100:.2f}%")

    # Interpretation
    empty_pct = 100.0 * n_zero / total
    print("\n--- INTERPRETATION ---")
    if empty_pct < 20.0:
        print(f"  Empty-sky crop fraction is {empty_pct:.1f}% -- LOW.")
        print("  Model rarely sees true empty sky. Hard negative mining (Task C change #3) is JUSTIFIED.")
    elif empty_pct < 40.0:
        print(f"  Empty-sky crop fraction is {empty_pct:.1f}% -- MODERATE.")
        print("  Some empty sky present. Hard negative mining may still help.")
    else:
        print(f"  Empty-sky crop fraction is {empty_pct:.1f}% -- HIGH.")
        print("  Model already sees plenty of empty sky. Hallucination is not a training-data issue.")

    # Save results
    os.makedirs("logs", exist_ok=True)
    result_file = "logs/24_fg_fraction_hist_results.txt"
    with open(result_file, "w") as f:
        f.write("FOREGROUND FRACTION HISTOGRAM\n")
        f.write(f"n_samples={args.n_samples}, crop_size={args.crop_size}, seed={args.seed}\n\n")
        f.write(f"Bucket | Count | %\n")
        for label, count in [
            ("0%  (empty)", n_zero),
            ("0-1%  (trace)", n_tiny),
            ("1-5%  (sparse)", n_low),
            ("5-20%  (moderate)", n_med),
            (">20%  (dense)", n_high),
        ]:
            f.write(f"{label}: {count} ({100.0*count/total:.1f}%)\n")
        f.write(f"\nMean: {fg_arr.mean()*100:.2f}% | Median: {np.median(fg_arr)*100:.2f}% | Std: {fg_arr.std()*100:.2f}%\n")
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
