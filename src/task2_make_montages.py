import json, os, random
import numpy as np
import cv2

HN_DIR = "data/hard_negatives_v1"
OUT_F  = "reports/hn_contamination_montage.png"
OUT_C  = "reports/hn_clean_montage.png"
THUMB  = 128
COLS   = 5

d       = json.load(open("reports/hn_contamination_manifest.json"))
results = d["results"]

def make_montage(paths, out):
    n = len(paths)
    if not n:
        print(f"  No paths for {out}")
        return
    rows   = (n + COLS - 1) // COLS
    canvas = np.zeros((rows * THUMB, COLS * THUMB, 3), np.uint8)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            print(f"  Could not read {p}")
            continue
        img = cv2.resize(img, (THUMB, THUMB))
        r, c = divmod(i, COLS)
        canvas[r*THUMB:(r+1)*THUMB, c*THUMB:(c+1)*THUMB] = img
    ok = cv2.imwrite(out, canvas)
    print(f"  imwrite {out} -> {ok}, shape={canvas.shape}")

flagged_paths = [os.path.join(HN_DIR, r["tile_file"]) for r in results if r["flagged"]][:30]
clean_paths   = [os.path.join(HN_DIR, r["tile_file"]) for r in results if not r["flagged"]]

print(f"Flagged tiles for montage : {len(flagged_paths)}")
print(f"Clean tiles available      : {len(clean_paths)}")

make_montage(flagged_paths, OUT_F)

random.seed(99)
sample_clean = random.sample(clean_paths, min(30, len(clean_paths)))
make_montage(sample_clean, OUT_C)

print("Done.")
