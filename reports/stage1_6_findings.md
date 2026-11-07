# Stage 1.6 — Expert 2's Three Checks: Findings & Decisions

**Date:** 2026-09-19  |  **Kaggle Score (Current Best):** 0.34  |  **Status:** Complete

---

## Task 1: Public Leaderboard Check

> **Skipped** (human action, per user request).  
> Current Kaggle public score: **0.34** (U-Net `prob_avg` TTA, `th=0.45`).

---

## Task 2: Complementary-Error Check (U-Net vs Mask R-CNN)

### Goal

Determine whether Mask R-CNN catches solar filaments that the U-Net misses on the 115-image held-out pool (769 total GT filaments). If yes, Stage 2 ensembling is viable.

### Results

| Metric | U-Net (prob_avg TTA) | Mask R-CNN |
|:---|:---:|:---:|
| True Positives (TP) | **485** | 397 |
| False Positives (FP) | **385** | 657 |
| False Negatives (FN) | **284** | 372 |
| Precision | **0.557** | 0.377 |
| Recall | **0.631** | 0.516 |


### Complementary Analysis

- **Complementary filaments** (U-Net missed, Mask R-CNN caught): **54**  
  - Recovery percentage: **19.01%** of U-Net's 284 FNs  
- **Double-covered filaments** (caught by both): 343 (86.40% of MRC-NN TPs)  
  - 86.4% of Mask R-CNN's TPs are already captured by U-Net  

### Decision Rule

| Threshold | Rule |
|:---:|:---|
| > 40 | Stage 2 viable — worth trying with strict confidence gating |
| 15–40 | Marginal — test only with very strict gating |
| < 15 | Stage 2 dead — drop ensemble |

> **Verdict:** `STAGE 2 VIABLE (Count > 40: worth trying with strict confidence gating)`

### Interpretation

- **54 complementary filaments** recovered. Stage 2 ensemble is viable.  
- However, Mask R-CNN has 657 FPs (vs U-Net's 346). Any ensemble must gate by confidence score > 0.70 + IoU non-overlap check to avoid importing false positives.  
- Net gain in recall will be marginal (~19%) but may push Kaggle score from 0.34 → ~0.35–0.36.  

---

## Task 3: Post-Processing Re-Sweep on TTA-Averaged Maps

### Goal

The locked config (`peak=55, prune=15, k=3, DIST_L1, th=0.45`) was tuned on single-view maps. TTA probability averaging changes map statistics. Re-swept post-processing on the 115-pool using cached TTA maps.

### Sweep Results

| | Config | Strict PQ | Delta vs Baseline | TP | FP | FN |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| Locked Baseline | peak=55, prune=15, k=3, L1, th=0.45 | **0.3967** | 0.0000 | — | — | — |
| **Best Found** | peak=65, prune=5, k=3, DIST_L2, th=0.35 | **0.4066** | **+0.0099** | 510 | 362 | 259 |


### Top 5 Configurations

| Rank | PQ | Peak | Prune | Kernel | DT | Thresh | Precision | Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.4066 | 65 | 5 | 3 | DIST_L2 | 0.35 | 0.585 | 0.663 |
| 2 | 0.4066 | 65 | 10 | 3 | DIST_L2 | 0.35 | 0.585 | 0.663 |
| 3 | 0.4066 | 65 | 15 | 3 | DIST_L2 | 0.35 | 0.585 | 0.663 |
| 4 | 0.4061 | 65 | 15 | 3 | DIST_L1 | 0.35 | 0.584 | 0.663 |
| 5 | 0.4061 | 65 | 5 | 3 | DIST_L1 | 0.35 | 0.584 | 0.663 |

### Decision Rule

| Threshold | Rule |
|:---:|:---|
| Local PQ > 0.4200 | Update config, generate new submission CSV |
| Local PQ ≤ 0.4200 | Keep current locked config |

> **Verdict:** `IMPROVEMENT FOUND (PQ=0.4066 vs baseline 0.3967, delta=+0.0099). Generate new submission with best config!`

---

## Summary & Next Actions

| Stage | Finding | Action |
|:---|:---|:---|
| Task 2 — Ensemble | 54 complementary filaments (19% FN recovery) | Stage 2 viable. Prototype gated ensemble with MRC-NN conf > 0.70 |
| Task 3 — Post-Proc Sweep | Best PQ: 0.4066 (≤ 0.4200) | Keep current locked config. Submission stands at 0.34 |

