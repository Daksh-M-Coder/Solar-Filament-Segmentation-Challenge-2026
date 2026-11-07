# Solar Filament Segmentation

Deep learning pipeline for detecting and segmenting solar filaments using H-Alpha solar imagery and MAGFiLO annotations.

---

## 📁 Repository Structure

```
solar-filament-segmentation/
│
├── data/                          # Raw and processed data
│   ├── raw/                       # The original H-Alpha images and MAGFiLO masks
│   │   ├── train/
│   │   ├── test/
│   │   └── MAGFiLO_annotations/
│   ├── processed/                 # Pre-processed data (resized, normalized, tiles)
│   └── external/                  # External datasets
│
├── notebooks/                     # Jupyter notebooks for exploration and submission
│   ├── 01_EDA_and_Visualization.ipynb
│   ├── 02_Training_Pipeline.ipynb
│   ├── 03_Inference_and_Submission.ipynb
│   └── 04_Self_Evaluation.ipynb   # Model evaluation and metric verification
│
├── src/                           # Modular source code
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset class for loading images/masks
│   ├── transforms.py              # Data augmentations (Albumentations / torchvision)
│   ├── model.py                   # Model architectures (U-Net, DeepLabV3+, etc.)
│   ├── loss.py                    # Loss functions (Dice, BCE, Focal, Combined)
│   ├── train.py                   # Script to train models
│   ├── predict.py                 # Script to run inference on test set
│   └── utils.py                   # Utilities (RLE encode/decode, metrics, plotting)
│
├── configs/                       # Configuration files (YAML)
│   └── config.yaml                # Hyperparameters, paths, model settings
│
├── models/                        # Saved model checkpoints and weights
│   └── best_model.pth
│
├── submissions/                   # Final submission CSV files
│   └── submission_v1.csv
│
├── reports/                       # Technical report and generated figures
│   └── figures/
│
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Create and activate a virtual environment (Python 3.10+ recommended):

```bash
# Using venv
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Place your dataset inside `data/raw/`:
- `data/raw/train/`: Training H-Alpha solar images
- `data/raw/test/`: Test set solar images
- `data/raw/MAGFiLO_annotations/`: Filament masks and annotations

### 3. Training

Update hyperparameters in [configs/config.yaml](file:///c:/Users/daksh/Programmer/Learning/COMP%20TECH%20SKILL/Kaggle%20Projects/01%20Solar%20Filament/configs/config.yaml), then run:

```bash
python -m src.train --config configs/config.yaml
```

### 4. Inference & Submission

Generate test predictions and submission files:

```bash
python -m src.predict --config configs/config.yaml
```
