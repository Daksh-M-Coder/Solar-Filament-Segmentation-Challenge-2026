# Audit 2: Temporal Leakage & Split Analysis

## Decision: **Case B (Temporal PQ 0.30 - 0.38: Moderate leakage; Kaggle 0.32 reflects true out-of-distribution performance. Proceed to Stage 1)**

### 1. Date Range & Proximity
- **Train Images Range**: `2011-01-09` to `2022-08-03` (1039 images)
- **Validation Images Range**: `2011-01-28` to `2021-12-31` (115 images)
- **Median Gap to Nearest Train Image**: `0.0 days`
- **Min Gap to Nearest Train Image**: `0.0 days`
- **25th - 75th Percentile Range**: `0.0` to `0.5 days`

### 2. Year Distribution
| Year | Train Images | Val Images | Val Fraction |
|:---:|:---:|:---:|:---:|
| 2011 | 139 | 11 | 7.33% |
| 2012 | 146 | 15 | 9.32% |
| 2013 | 136 | 20 | 12.82% |
| 2014 | 148 | 13 | 8.07% |
| 2015 | 130 | 10 | 7.14% |
| 2016 | 115 | 12 | 9.45% |
| 2017 | 97 | 21 | 17.80% |
| 2018 | 40 | 2 | 4.76% |
| 2019 | 22 | 2 | 8.33% |
| 2020 | 11 | 3 | 21.43% |
| 2021 | 36 | 6 | 14.29% |
| 2022 | 19 | 0 | 0.00% |

### 3. Temporal Split Performance (2020+ Images)
- Evaluated Images: 75
- **Strict PQ**: **`0.3395`**
- **True Positives (TP)**: `149`
- **False Positives (FP)**: `125`
- **False Negatives (FN)**: `126`
