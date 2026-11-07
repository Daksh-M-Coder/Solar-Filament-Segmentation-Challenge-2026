# Stage 0 — Extended Free Sweep Results

## 1. Optimal Production Configuration
- **`peak_min_distance`**: `55`
- **`prune_threshold`**: `15`
- **`score_threshold`**: `0.45`
- **`kernel_size`**: `3`
- **`dist_transform`**: `DIST_L1`
- **Final Local Strict PQ**: **`0.4162`**

## 2. Production Performance Metrics (20 Validation Images)
| Metric | Value |
|:---|:---|
| Strict PQ | **0.4162** |
| True Positives (TP) | 107 |
| False Positives (FP) | 75 |
| False Negatives (FN) | 55 |
| Precision | 0.5879 |
| Recall | 0.6605 |
| Mean Instances / Image | 9.10 |
| Mean Branch Points / Comp | 2.70 |

## 3. Single Parameter Sweep Tables (Task 0.1)

### peak_min_distance
| Value | Strict PQ | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|
| 25 | 0.3279 | 88 | 128 | 74 |
| 30 | 0.3466 | 91 | 114 | 71 |
| 35 | 0.3679 | 95 | 102 | 67 |
| 40 | 0.3773 | 97 | 94 | 65 |
| 45 | 0.3841 | 98 | 88 | 64 |
| 50 | 0.3963 | 99 | 84 | 63 |

### prune_threshold
| Value | Strict PQ | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|
| 10 | 0.2938 | 81 | 164 | 81 |
| 15 | 0.2972 | 82 | 157 | 80 |
| 20 | 0.2972 | 82 | 157 | 80 |
| 25 | 0.2972 | 82 | 157 | 80 |
| 30 | 0.2972 | 82 | 157 | 80 |

### score_threshold
| Value | Strict PQ | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|
| 0.45 | 0.2888 | 82 | 172 | 80 |
| 0.5 | 0.2938 | 81 | 164 | 81 |
| 0.55 | 0.2961 | 81 | 160 | 81 |
| 0.6 | 0.2885 | 78 | 160 | 84 |
| 0.65 | 0.2865 | 78 | 157 | 84 |
| 0.7 | 0.2811 | 77 | 160 | 85 |

### kernel_size
| Value | Strict PQ | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|
| 3 | 0.2911 | 82 | 163 | 80 |
| 5 | 0.2938 | 81 | 164 | 81 |
| 7 | 0.2871 | 77 | 152 | 85 |
| 9 | 0.1819 | 54 | 148 | 108 |

### dist_transform
| Value | Strict PQ | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|
| DIST_L1 | 0.2589 | 76 | 187 | 86 |
| DIST_L2 | 0.2938 | 81 | 164 | 81 |

## 4. Joint Sweeps (Task 0.2)
### Set A (Peak x Prune)
| Peak | Prune | Strict PQ | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|
| 45 | 10 | 0.3841 | 98 | 88 | 64 |
| 45 | 15 | 0.3870 | 99 | 85 | 63 |
| 45 | 20 | 0.3870 | 99 | 85 | 63 |
| 50 | 10 | 0.3963 | 99 | 84 | 63 |
| 50 | 15 | 0.3992 | 100 | 81 | 62 |
| 50 | 20 | 0.3992 | 100 | 81 | 62 |
| 55 | 10 | 0.4001 | 101 | 80 | 61 |
| 55 | 15 | 0.4013 | 101 | 79 | 61 |
| 55 | 20 | 0.4013 | 101 | 79 | 61 |

### Set B (Score Threshold)
| Score Thresh | Strict PQ | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|
| 0.45 | 0.4087 | 104 | 80 | 58 |
| 0.50 | 0.4013 | 101 | 79 | 61 |
| 0.55 | 0.4003 | 99 | 77 | 63 |
| 0.60 | 0.3972 | 97 | 74 | 65 |
| 0.65 | 0.3962 | 97 | 75 | 65 |

### Set C (Kernel Size)
| Kernel Size | Strict PQ | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|
| 3 | 0.4140 | 106 | 77 | 56 |
| 5 | 0.4087 | 104 | 80 | 58 |
| 7 | 0.3934 | 98 | 77 | 64 |

### Set D (Distance Transform)
| Distance Transform | Strict PQ | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|
| DIST_L1 | 0.4162 | 107 | 75 | 55 |
| DIST_L2 | 0.4140 | 106 | 77 | 56 |

