# FP Bucket Split Report

**Checkpoint:** `models/unet_cldice_v2_best.pth`  
**TTA:** True (prob_avg)  
**Pool:** 115 images  
**Locked config:** peak=55, prune=15, k=3, DIST_L1, th=0.45, min_area=300

## Part A — FP instances from live validation (115-pool)

Total FP instances: **360**

| Bucket | Count | % of total FPs | Mean Area (px) |
|--------|------:|---------------:|---------------:|
| (a) IoU=0.000 | 158 | 43.9% | 979 |
| (b) 0.000<IoU<=0.30 | 110 | 30.6% | 1068 |
| (c) 0.30<IoU<=0.45 | 64 | 17.8% | 1891 |
| (d) 0.45<IoU<=0.48 | 16 | 4.4% | 1947 |
| (e) 0.48<IoU<=0.50 | 12 | 3.3% | 1397 |

## Part B — Mined HN pool breakdown (top-500 saved tiles)

Total HN tiles: **500**  
*Note: by definition all mined HN have best_iou=0.000 (IoU=0 against all GT).*

| Bucket | Count | % of total HN | Mean Area (px) |
|--------|------:|--------------:|---------------:|
| (a) IoU=0.000 | 500 | 100.0% | 1942 |
| (b) 0.000<IoU<=0.30 | 0 | 0.0% | 0 |
| (c) 0.30<IoU<=0.45 | 0 | 0.0% | 0 |
| (d) 0.45<IoU<=0.48 | 0 | 0.0% | 0 |
| (e) 0.48<IoU<=0.50 | 0 | 0.0% | 0 |

## Notes

- FP bucketing uses strict PQ IoU=0.5 matching, same as the validator.
- Bucket (a) are pure false positives with zero overlap with any GT filament.
- Buckets (d) and (e) are near-misses — almost matched but fell just below threshold.
- HN pool was mined separately from the training set, so all entries land in bucket (a) by construction.
