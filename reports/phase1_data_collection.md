# Phase 1: Data Collection & Understanding

## Dataset: Next Day Wildfire Spread

- **Source**: Google Research / Stanford — Kaggle
- **Type**: Satellite imagery (Landsat-8) + weather data (GRIDMET)
- **Format**: TFRecord shards
- **Size**: ~2.08 GB, ~15,000 training patches

## Data Structure

Each sample is a **64×64 pixel patch** with:
- **12 input channels**: NDVI, elevation, wind (th, vs), temperature (tmmn, tmmx), humidity (sph), precipitation (pr), drought index (pdsi), energy release (erc), population density, previous fire mask
- **1 label**: FireMask with values -1 (unknown), 0 (no fire), 1 (fire)

## Splits
| Split | Shards | Samples |
|-------|--------|---------|
| Train | 15 | ~15,000 |
| Eval  | 2  | ~2,000  |
| Test  | 2  | ~2,000  |

## Key Observations
- Highly imbalanced: fire pixels are <2% of known pixels
- Unknown pixels (-1) must be masked during loss computation
- Satellite features have very different scales → normalization required
