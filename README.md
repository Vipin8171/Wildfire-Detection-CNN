# 🔥 Wildfire Spread Detection Using Satellite Imagery

AI-powered **next-day wildfire spread prediction** using 12-channel Landsat-8 satellite data with an Attention U-Net deep learning model.

## 📋 Project Overview

| Aspect | Detail |
|---|---|
| **Task** | Pixel-level fire spread segmentation (64×64 patches) |
| **Dataset** | [Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) (~15K samples) |
| **Model** | Attention U-Net with SE blocks |
| **Input** | 12 satellite + weather channels |
| **Output** | Binary fire mask (will-burn / won't-burn) |

## 🛰️ 12 Input Features

| # | Feature | Source | Description |
|---|---|---|---|
| 1 | NDVI | Landsat-8 | Vegetation index |
| 2 | Elevation | SRTM | Terrain height (m) |
| 3 | th | GRIDMET | Wind direction (°) |
| 4 | vs | GRIDMET | Wind speed (m/s) |
| 5 | tmmn | GRIDMET | Min temperature (K) |
| 6 | tmmx | GRIDMET | Max temperature (K) |
| 7 | sph | GRIDMET | Specific humidity (kg/kg) |
| 8 | pr | GRIDMET | Precipitation (mm) |
| 9 | pdsi | GRIDMET | Palmer Drought Index |
| 10 | erc | GRIDMET | Energy Release Component |
| 11 | Population | GPWv4 | Population density |
| 12 | PrevFireMask | MODIS | Previous day fire |

## 🏗️ Project Structure

```
Wildfire-Detection-CNN/
├── src/
│   ├── data_loader.py      # TFRecord → PyTorch DataLoader
│   ├── models.py            # Attention U-Net + loss functions
│   ├── train.py             # Training loop with metrics
│   ├── evaluate.py          # Test evaluation + visualizations
│   ├── predict.py           # Inference + reports
│   └── preprocessing.py     # Data quality & feature engineering
├── webapp/
│   ├── app.py               # Flask web application
│   └── templates/
│       ├── index.html        # Upload & predict UI
│       └── about.html        # Project information
├── checkpoints/              # Saved model weights
├── results/                  # Evaluation outputs
├── reports/                  # Phase reports
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python src/download_dataset.py

# 3. Train model
python src/train.py --epochs 50 --batch_size 32

# 4. Evaluate
python src/evaluate.py

# 5. Run predictions
python src/predict.py

# 6. Launch webapp
python webapp/app.py
```

## 🧠 Model Architecture

**Attention U-Net** with:
- **Encoder**: 4 stages (64→128→256→512 filters) with MaxPool
- **Bottleneck**: Conv blocks + Squeeze-and-Excitation attention
- **Decoder**: 4 stages with transposed convolutions + attention gates
- **Loss**: Combined BCE + Dice loss with masked unknown pixels
- **Class weighting**: Handles severe fire/no-fire pixel imbalance

## 📊 Training Details

- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: Cosine annealing
- Augmentation: Random flips + 90° rotations
- Early stopping: Patience = 10 epochs
- Mixed precision: AMP on CUDA
