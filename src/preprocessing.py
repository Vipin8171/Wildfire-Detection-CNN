"""
Preprocessing Utilities
========================
Feature engineering, patch augmentation, and data quality checks
for satellite wildfire data.
"""

import numpy as np
from src.data_loader import INPUT_FEATURES, PATCH_SIZE


def check_data_quality(X, y):
    """Check for NaN, inf, and value ranges."""
    report = {}
    for i, feat in enumerate(INPUT_FEATURES):
        ch = X[:, i]
        report[feat] = {
            "nan_count": int(np.isnan(ch).sum()),
            "inf_count": int(np.isinf(ch).sum()),
            "min": float(np.nanmin(ch)),
            "max": float(np.nanmax(ch)),
            "mean": float(np.nanmean(ch)),
            "std": float(np.nanstd(ch)),
        }
    report["FireMask"] = {
        "unique_values": sorted(np.unique(y).tolist()),
        "fire_pixels": int((y == 1).sum()),
        "no_fire_pixels": int((y == 0).sum()),
        "unknown_pixels": int((y == -1).sum()),
        "total_pixels": int(y.size),
        "fire_ratio": float((y == 1).sum() / ((y != -1).sum() + 1e-8)),
    }
    return report


def add_derived_features(X):
    """
    Optionally compute derived features from the 12 base channels.
    Returns: X_extended with additional channels appended.
    
    Currently returns X unchanged — extend here if you want:
      - Temperature range (tmmx - tmmn)
      - Fire Weather Index approximation
      - Vegetation dryness (NDVI × PDSI interaction)
    """
    # Example: temperature range
    # tmmn_idx, tmmx_idx = INPUT_FEATURES.index("tmmn"), INPUT_FEATURES.index("tmmx")
    # temp_range = X[:, tmmx_idx:tmmx_idx+1] - X[:, tmmn_idx:tmmn_idx+1]
    # X = np.concatenate([X, temp_range], axis=1)
    return X


def clip_outliers(X, percentile=99.5):
    """Clip extreme values per channel."""
    X_clipped = X.copy()
    for c in range(X.shape[1]):
        ch = X_clipped[:, c]
        valid = ch[~np.isnan(ch)]
        if len(valid) > 0:
            low = np.percentile(valid, 100 - percentile)
            high = np.percentile(valid, percentile)
            X_clipped[:, c] = np.clip(ch, low, high)
    return X_clipped
