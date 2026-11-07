# Task 0: Pre-Flight Data & Artifact Inspection Report

## 0.1 JSON Inspection Results
- **Dataset File:** `data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`
- **Total Images:** 1,154
- **Total Annotations:** 8,199
- **Categories:** Left (1), Right (2), Unidentifiable (3), Ambiguous (4)
- **Annotation Keys:** `['segmentation', 'area', 'iscrowd', 'spine', 'image_id', 'bbox', 'category_id', 'id']`
- **Spine Key Presence:** **`True`** (contains raw coordinates list of the true spine polyline, e.g., 54 coordinates for sample entry).
- **Skeleton / Centerline Keys:** `False`.
- **Decision:** Following Task 0 Decision Rules, since `spine` exists and is non-empty, we enable the **True Spine head**.

## 0.2 Submission Format Inspection Results
- **Sample File Examined:** `archive/Mask R-CNN/submissions/submission_th0.15.csv`
- **Columns:** `filament_id`, `segmentation_rle`
- **Index:** No extraneous index column (`index=False`).
- **Format:** Exact match with Kaggle requirements:
  - `filament_id`: `{image_stem}_{instance_number}`
  - `segmentation_rle`: Fortran-contiguous string encoding of the binary instance mask.

## 0.3 Image Artifact Inspection Results
- **Images Examined:** 25 randomly sampled 2048x2048 solar images.
- **Artifact Discontinuity Check:** Checked row and column profiles across quarters and midpoints (`h/4, h/2, 3h/4`, etc.) using MAD gradient thresholding.
- **Findings:** A total of 636 repeating coordinate discontinuities (> 3 images) were identified near disk borders and quadrant boundaries (`row_1024`, `col_1024`, `col_121`, `col_1926`).
- **Impact on Task 5 Post-processing:** Shape filtering, area thresholding (`min_area=300`), and morphological cleaning are incorporated to filter out fine linear artifact edges.
