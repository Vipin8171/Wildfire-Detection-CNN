# Phase 2: Exploratory Data Analysis

## Feature Statistics
- **NDVI**: Range [-1, 1], mean ~0.30 — vegetation health indicator
- **Elevation**: Range 0–4000+ m — topographic factor affecting fire spread
- **Temperature**: tmmn ~250–310K, tmmx ~260–330K — key fire risk factor
- **Wind**: Direction (th) 0–360°, Speed (vs) 0–15+ m/s — drives fire spread
- **Humidity**: sph 0–0.02 kg/kg — low humidity = higher fire risk
- **Precipitation**: pr 0–50+ mm — rain suppresses fire
- **Drought**: pdsi -10 to +10 — negative = dry conditions
- **ERC**: 0–100+ — energy available for fire combustion
- **Population**: 0–10,000+ persons/km² — human impact factor
- **PrevFireMask**: Binary 0/1 — strongest predictor of next-day fire

## Class Imbalance
- Fire pixels: ~1-2% of all known pixels
- No-fire pixels: ~98-99%
- Unknown pixels: varies per patch
- **Solution**: Positive class weighting (~20-50x) in BCE loss + Dice loss

## Correlations
- PrevFireMask → FireMask: strongest correlation (fire continues)
- High ERC + Low humidity → higher fire probability
- Low NDVI (dry vegetation) + high temperature → fire risk
