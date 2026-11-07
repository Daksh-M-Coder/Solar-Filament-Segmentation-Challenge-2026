# Evidence Index for Solar Filament Competition Report
# Maintained alongside reports/final_report_draft.md
# Rule: Every numeric claim in the final report must trace to an entry in this index.

| Experiment Name | Date | Log File | Key Number | Verdict | Section Fed |
|---|---|---|---|---|---|
| Mask R-CNN baseline | Past | logs/01_maskrcnn_baseline.log | Kaggle PQ: 0.07 | Rejected | Sec 2, Sec 8 |
| Tiled inference (Mask R-CNN) | Past | logs/02_tiled_maskrcnn.log | Kaggle PQ: 0.12 | Adopted (concept) | Sec 2, Sec 8 |
| Threshold tuning | Past | logs/03_thresh_tuning.log | Kaggle PQ: 0.18 | Adopted | Sec 2, Sec 8 |
| Retrain + hard negatives | Past | logs/04_retrain_hardneg.log | Kaggle PQ: 0.20 | Adopted | Sec 2, Sec 8 |
| U-Net refiner | Past | logs/05_unet_refiner.log | Kaggle PQ: 0.22 | Adopted | Sec 2, Sec 8 |
| U-Net pivot (standalone) | Past | logs/06_unet_standalone.log | Kaggle PQ: 0.26 | Adopted | Sec 2, Sec 8 |
| Post-processing sweep | Past | logs/07_postproc_sweep.log | Local PQ: 0.32 | Adopted | Sec 2, Sec 3 |
| TTA mask-vote (v1) | Past | logs/08_tta_maskvote.log | Local PQ: 0.33 | Replaced | Sec 2, Sec 7 |
| TTA prob-averaging (corrected) | Past | logs/09_tta_probavg.log | Local PQ: 0.34 (+0.011) | Adopted | Sec 2, Sec 7 |
| Ensemble with Mask R-CNN | Past | logs/10_ensemble_mrnn.log | Delta: +0.0009 / +0.0026 | Rejected | Sec 2, Sec 5 |
| clDice v1 (broken target) | Past | logs/11_cldice_v1.log | Regression, skel_true broken | Rejected | Sec 2, Sec 5 |
| Re-sweep on cached maps | 2026-09-25 | logs/20_cached_sweep.log | Apparent +0.0099 (cached artifact) | Rejected | Sec 2, Sec 3, Sec 5 |
| Live validation of swept config | 2026-09-25 | logs/21_validate_swept_config_115.csv | Local PQ: 0.4123 vs 0.4154 | Rejected | Sec 2, Sec 3 |
| min_area sweep | 2026-09-25 | logs/22_min_area_sweep.log | 300 optimal (flat across 100-800) | Adopted (kept 300) | Sec 2, Sec 3 |
| Tile-seam prob averaging (prob_avg_seam) | 2026-09-25 | logs/16_val_seam.log | Local PQ: 0.4078 vs 0.4154 | Rejected | Sec 2, Sec 5 |
| clDice v2 30-epoch training | 2026-09-25 | logs/25_train_cldice_v2.log | Ep 25 Val PQ: 0.4212 vs 0.4154 baseline | Adopted | Sec 2, Sec 7 |
| clDice v2 Verification Run 1 | 2026-09-25 | logs/26_cldice_verify_run1.csv | Strict PQ: 0.4212 (TP=523, FP=360, FN=246) | Confirmed Real | Sec 2, Sec 8 |
| clDice v2 Verification Run 2 | 2026-09-25 | logs/26_cldice_verify_run2.csv | Strict PQ: 0.4212 (TP=523, FP=360, FN=246) | Confirmed Real | Sec 2, Sec 8 |
| Old Baseline Control Run | 2026-09-25 | logs/26_oldref.csv | Strict PQ: 0.4154 (TP=510, FP=346, FN=259) | Control Baseline | Sec 2, Sec 8 |
| Multi-threshold clDice Submissions | 2026-09-25 | submissions/submission_cldice_th*.csv | 6 CSVs generated, zero overlaps verified | Ready for Kaggle | Sec 2, Sec 8 |
| Foreground-fraction diagnostic | 2026-09-25 | logs/24_fg_fraction_hist_results.txt | 23.7% empty, 99.0% <5% FG, mean 0.92% | Diagnostic | Sec 2, Sec 6 |
| FP dump diagnostic | 2026-09-25 | logs/23_fp_dump.log | Top-20 largest FPs visualized | Diagnostic | Sec 2, Sec 6 |
| Filament Area & Contrast Diagnostic | 2026-09-26 | data/filament_oversampling_targets.json | Q1 Area: 671.2px, Q1 Contrast: 11.87, 255 small+faint | Diagnostic | Sec 2, Sec 6 |
| Hard-Negative Pool Extraction | 2026-09-26 | logs/27_extract_hard_negatives.log | 1,539 instances across 671 images; top 500 saved | Pool Ready | Sec 2, Sec 7 |
