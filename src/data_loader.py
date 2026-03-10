"""
Data Loader for Next Day Wildfire Spread Dataset
================================================
Loads TFRecord satellite data (Landsat-8 + weather + terrain)
into PyTorch tensors for fire spread segmentation.

Dataset: 64x64 patches, 12 input channels, binary fire mask label.
Features: NDVI, elevation, th, vs, tmmn, tmmx, sph, pr, pdsi, erc, population, PrevFireMask
Label:    FireMask  (-1 = unknown, 0 = no fire, 1 = fire)
"""

import os
import glob
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader

# ──────────────────── constants ────────────────────
DATA_DIR = r"C:\Users\tvipi\.cache\kagglehub\datasets\fantineh\next-day-wildfire-spread\versions\2"

INPUT_FEATURES = [
    "NDVI", "elevation", "th", "vs", "tmmn",
    "tmmx", "sph", "pr", "pdsi", "erc",
    "population", "PrevFireMask",
]
OUTPUT_FEATURES = ["FireMask"]
PATCH_SIZE = 64
NUM_CHANNELS = len(INPUT_FEATURES)  # 12

# Human-readable info for the webapp / reports
FEATURE_INFO = {
    "NDVI":         {"full_name": "Normalized Difference Vegetation Index", "source": "Landsat-8 (VIIRS)", "unit": "index (-1 to 1)"},
    "elevation":    {"full_name": "Terrain Elevation",                     "source": "SRTM",              "unit": "meters"},
    "th":           {"full_name": "Wind Direction",                        "source": "GRIDMET",            "unit": "degrees from North"},
    "vs":           {"full_name": "Wind Speed",                            "source": "GRIDMET",            "unit": "m/s"},
    "tmmn":         {"full_name": "Minimum Temperature",                   "source": "GRIDMET",            "unit": "K"},
    "tmmx":         {"full_name": "Maximum Temperature",                   "source": "GRIDMET",            "unit": "K"},
    "sph":          {"full_name": "Specific Humidity",                     "source": "GRIDMET",            "unit": "kg/kg"},
    "pr":           {"full_name": "Precipitation",                         "source": "GRIDMET",            "unit": "mm"},
    "pdsi":         {"full_name": "Palmer Drought Severity Index",         "source": "GRIDMET",            "unit": "index"},
    "erc":          {"full_name": "Energy Release Component",              "source": "GRIDMET",            "unit": "index"},
    "population":   {"full_name": "Population Density",                    "source": "GPWv4",              "unit": "persons/km²"},
    "PrevFireMask": {"full_name": "Previous Day Fire Mask",                "source": "MODIS",              "unit": "binary (0/1)"},
    "FireMask":     {"full_name": "Next Day Fire Mask (label)",            "source": "MODIS",              "unit": "-1/0/1"},
}

# Pre-computed approximate stats (mean, std) for each feature across the training set.
# These will be refined on first load; initial values from the dataset paper.
CHANNEL_STATS = {
    "NDVI":         (0.30,   0.15),
    "elevation":    (1200.0, 800.0),
    "th":           (220.0,  50.0),
    "vs":           (3.5,    1.8),
    "tmmn":         (280.0,  12.0),
    "tmmx":         (295.0,  12.0),
    "sph":          (0.006,  0.004),
    "pr":           (1.0,    3.0),
    "pdsi":         (-1.5,   2.5),
    "erc":          (40.0,   25.0),
    "population":   (100.0,  500.0),
    "PrevFireMask": (0.02,   0.14),
}


# ──────────────────── TFRecord parsing ────────────────────
def _get_feature_description():
    """Build the TFRecord feature description dict."""
    desc = {}
    for feat in INPUT_FEATURES + OUTPUT_FEATURES:
        desc[feat] = tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32)
    return desc


def _parse_tfrecord(serialised):
    """Parse a single TFRecord example into (inputs, label) numpy arrays."""
    desc = _get_feature_description()
    example = tf.io.parse_single_example(serialised, desc)

    inputs = np.stack(
        [example[f].numpy().reshape(PATCH_SIZE, PATCH_SIZE) for f in INPUT_FEATURES],
        axis=0,
    )                                       # (12, 64, 64)
    label = example["FireMask"].numpy().reshape(PATCH_SIZE, PATCH_SIZE)  # (64, 64)
    return inputs, label


def _get_shard_files(split):
    """Find TFRecord shard files for a given split."""
    pattern = os.path.join(DATA_DIR, f"{split}", "*.tfrecord*")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(DATA_DIR, f"*{split}*")
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TFRecord files for split='{split}' in {DATA_DIR}")
    return files


def load_tfrecord_data(split="train", max_shards=None):
    """Load samples from a split into numpy arrays.

    Parameters
    ----------
    split : str  "train", "eval", or "test"
    max_shards : int or None  limit number of shards (for memory)

    Returns
    -------
    X : np.ndarray  shape (N, 12, 64, 64)  float32
    y : np.ndarray  shape (N, 64, 64)       float32   values in {-1, 0, 1}
    """
    files = _get_shard_files(split)
    if max_shards is not None:
        files = files[:max_shards]

    print(f"[data_loader] Loading {split}: {len(files)} shard(s) …")
    all_X, all_y = [], []
    for fpath in files:
        dataset = tf.data.TFRecordDataset(fpath)
        for raw in dataset:
            x, y = _parse_tfrecord(raw)
            all_X.append(x)
            all_y.append(y)

    X = np.stack(all_X).astype(np.float32)
    y = np.stack(all_y).astype(np.float32)
    print(f"[data_loader] {split}: X {X.shape}, y {y.shape}  "
          f"(fire pixels: {(y == 1).sum():,} / {y.size:,})")
    return X, y


# ──────────────────── Normalization ────────────────────
def compute_channel_stats_streaming(split="train", max_shards=3):
    """Compute per-channel mean/std by streaming through shards (low memory)."""
    files = _get_shard_files(split)[:max_shards]
    sums = np.zeros(NUM_CHANNELS, dtype=np.float64)
    sq_sums = np.zeros(NUM_CHANNELS, dtype=np.float64)
    counts = np.zeros(NUM_CHANNELS, dtype=np.float64)
    fire_count = 0
    known_count = 0

    for fpath in files:
        dataset = tf.data.TFRecordDataset(fpath)
        for raw in dataset:
            x, y = _parse_tfrecord(raw)  # (12,64,64), (64,64)
            for c in range(NUM_CHANNELS):
                ch = x[c]
                valid = ch[~np.isnan(ch)]
                sums[c] += valid.sum()
                sq_sums[c] += (valid ** 2).sum()
                counts[c] += len(valid)
            fire_count += (y == 1).sum()
            known_count += (y != -1).sum()

    means = (sums / np.maximum(counts, 1)).astype(np.float32)
    stds = np.sqrt(sq_sums / np.maximum(counts, 1) - means ** 2).astype(np.float32)
    stds = np.where(stds == 0, 1.0, stds)
    fire_ratio = float(fire_count / max(known_count, 1))
    return means, stds, fire_ratio


def compute_channel_stats(X):
    """Compute per-channel mean and std from in-memory array."""
    means, stds = [], []
    for c in range(X.shape[1]):
        ch = X[:, c]
        valid = ch[~np.isnan(ch)]
        means.append(valid.mean() if len(valid) else 0.0)
        stds.append(valid.std() if len(valid) else 1.0)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def normalize(X, means, stds):
    """Z-score normalization; NaN → 0."""
    stds = np.where(stds == 0, 1.0, stds)
    X_norm = (X - means[None, :, None, None]) / stds[None, :, None, None]
    X_norm = np.nan_to_num(X_norm, nan=0.0)
    return X_norm


# ──────────────────── Lazy TFRecord Dataset ────────────────────
class LazyTFRecordDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset that reads TFRecord files lazily.
    Builds an index of (file, offset) pairs but only parses on __getitem__.
    """

    def __init__(self, split, means, stds, augment=False, max_shards=None):
        self.means = means
        self.stds = stds
        self.augment = augment

        # Build index: list of (file_path, record_index_within_file)
        files = _get_shard_files(split)
        if max_shards is not None:
            files = files[:max_shards]

        self.index = []
        for fpath in files:
            count = sum(1 for _ in tf.data.TFRecordDataset(fpath))
            for i in range(count):
                self.index.append((fpath, i))
        print(f"[LazyDS] {split}: {len(self.index)} samples from {len(files)} shards")

        # Cache parsed records per file for speed
        self._cache = {}

    def _load_file(self, fpath):
        if fpath not in self._cache:
            records = []
            for raw in tf.data.TFRecordDataset(fpath):
                records.append(raw)
            self._cache[fpath] = records
            # Keep cache bounded (max 3 files at a time)
            if len(self._cache) > 3:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
        return self._cache[fpath]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, rec_idx = self.index[idx]
        records = self._load_file(fpath)
        x_np, y_np = _parse_tfrecord(records[rec_idx])

        # Normalize
        stds_safe = np.where(self.stds == 0, 1.0, self.stds)
        x_np = (x_np - self.means[:, None, None]) / stds_safe[:, None, None]
        x_np = np.nan_to_num(x_np, nan=0.0)

        x = torch.from_numpy(x_np.astype(np.float32))
        label = torch.from_numpy((y_np == 1).astype(np.float32))
        weight = torch.from_numpy((y_np != -1).astype(np.float32))

        if self.augment:
            if torch.rand(1) > 0.5:
                x = x.flip(-1); label = label.flip(-1); weight = weight.flip(-1)
            if torch.rand(1) > 0.5:
                x = x.flip(-2); label = label.flip(-2); weight = weight.flip(-2)
            k = torch.randint(0, 4, (1,)).item()
            if k:
                x = torch.rot90(x, k, [-2, -1])
                label = torch.rot90(label, k, [-2, -1])
                weight = torch.rot90(weight, k, [-2, -1])

        return x, label, weight


# ──────────────────── In-memory Dataset ────────────────────
class WildfireDataset(Dataset):
    """In-memory dataset (for small splits like eval/test)."""

    def __init__(self, X, y, means=None, stds=None, augment=False):
        if means is not None and stds is not None:
            X = normalize(X, means, stds)
        self.X = torch.from_numpy(X)
        label = (y == 1).astype(np.float32)
        weight = (y != -1).astype(np.float32)
        self.y = torch.from_numpy(label)
        self.w = torch.from_numpy(weight)
        self.augment = augment

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, label, weight = self.X[idx], self.y[idx], self.w[idx]
        if self.augment:
            if torch.rand(1) > 0.5:
                x = x.flip(-1); label = label.flip(-1); weight = weight.flip(-1)
            if torch.rand(1) > 0.5:
                x = x.flip(-2); label = label.flip(-2); weight = weight.flip(-2)
            k = torch.randint(0, 4, (1,)).item()
            if k:
                x = torch.rot90(x, k, [-2, -1])
                label = torch.rot90(label, k, [-2, -1])
                weight = torch.rot90(weight, k, [-2, -1])
        return x, label, weight


# ──────────────────── DataLoader factory ────────────────────
def get_dataloaders(batch_size=32, num_workers=0, train_shards=None):
    """Create train / val / test DataLoaders with memory-efficient loading.

    Parameters
    ----------
    train_shards : int or None  — limit training shards (default: all 15)

    Returns
    -------
    train_loader, val_loader, test_loader, info dict
    """
    # Compute stats from a few training shards (streaming, low memory)
    print("[data_loader] Computing channel statistics …")
    means, stds, fire_ratio = compute_channel_stats_streaming("train", max_shards=3)
    pos_weight = (1 - fire_ratio) / (fire_ratio + 1e-8)
    print(f"[data_loader] means: {means}")
    print(f"[data_loader] stds : {stds}")
    print(f"[data_loader] fire_ratio={fire_ratio:.5f}, pos_weight={pos_weight:.2f}")

    # Training: lazy loading (memory efficient)
    train_ds = LazyTFRecordDataset("train", means, stds, augment=True,
                                    max_shards=train_shards)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)

    # Eval & test: small enough to load in-memory
    X_val, y_val = load_tfrecord_data("eval")
    X_test, y_test = load_tfrecord_data("test")
    val_ds = WildfireDataset(X_val, y_val, means, stds, augment=False)
    test_ds = WildfireDataset(X_test, y_test, means, stds, augment=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)

    info = {
        "means": means,
        "stds": stds,
        "pos_weight": float(pos_weight),
        "fire_ratio": fire_ratio,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
    }
    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    train_loader, val_loader, test_loader, info = get_dataloaders(batch_size=16)
    print("\n=== Dataset Summary ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    x, label, w = next(iter(train_loader))
    print(f"\nSample batch → x: {x.shape}, label: {label.shape}, weight: {w.shape}")
