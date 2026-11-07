# Stage 1: TTA Mode Comparison Report

## Winner: **`PROB_AVG`**

### Comparison Table (115 Validation Images at Threshold 0.45)

| Metric | `mask_vote` (Strict Majority >=3/4) | `prob_avg` (Raw Probability Averaging) | Delta (prob_avg - mask_vote) |
|:---|:---:|:---:|:---:|
| **Strict PQ** | **0.4043** | **0.4154** | **+0.0111** |
| True Positives (TP) | 477 | 510 | +33 |
| False Positives (FP) | 336 | 346 | +10 |
| False Negatives (FN) | 292 | 259 | -33 |
| Mean Raw Prob Variance | 0.0001 | 0.0001 | +0.0000 |
| Unstable Tiles (>0.05 var) | 0 | 0 | +0 |
| Total Tiles Evaluated | 2875 | 2875 | 0 |
| Wall-Clock Time | 621.3s (10.4m) | 637.9s (10.6m) | +16.5s |

### Key Observations
- **Accuracy**: `prob_avg` achieved **0.4154** PQ vs `mask_vote` **0.4043** PQ.
- **Speed**: `prob_avg` completed in **637.9s** vs **621.3s** for `mask_vote` (post-processing executed once per tile instead of 4 times).
