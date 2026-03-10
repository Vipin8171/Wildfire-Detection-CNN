"""
Extract Sample Data from TFRecord
==================================
Extracts sample patches from the TFRecord dataset and saves them as .npy files
for easy testing in the webapp without needing TFRecord parsing.
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_tfrecord_data


def extract_samples(output_dir="data", num_samples=10):
    """
    Extract sample patches from test set and save as .npy files.
    
    Parameters
    ----------
    output_dir : str
        Directory to save sample .npy files
    num_samples : int
        Number of samples to extract (prefers samples with fire)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("[extract] Loading test data ...")
    X_test, y_test = load_tfrecord_data("test")
    
    # Find samples with most fire pixels (for interesting test cases)
    fire_counts = (y_test == 1).sum(axis=(1, 2))
    
    # Get samples with varying amounts of fire
    fire_ranges = [
        (100, 200, "low_fire"),      # 100-200 fire pixels
        (200, 500, "medium_fire"),   # 200-500 fire pixels
        (500, 2000, "high_fire"),    # 500+ fire pixels
    ]
    
    saved_count = 0
    
    for min_fire, max_fire, label in fire_ranges:
        valid_idx = np.where((fire_counts >= min_fire) & (fire_counts < max_fire))[0]
        
        if len(valid_idx) > 0:
            # Randomly select one from this range
            idx = np.random.choice(valid_idx)
            patch = X_test[idx].astype(np.float32)
            
            filename = os.path.join(output_dir, f"sample_{label}_{saved_count:02d}.npy")
            np.save(filename, patch)
            
            fire_pixels = fire_counts[idx]
            print(f"  ✓ Saved {filename} ({int(fire_pixels)} fire pixels)")
            saved_count += 1
    
    # Also save some no-fire examples
    no_fire_idx = np.where(fire_counts < 10)[0]
    if len(no_fire_idx) > 0:
        for i in range(min(3, len(no_fire_idx))):
            idx = no_fire_idx[i]
            patch = X_test[idx].astype(np.float32)
            
            filename = os.path.join(output_dir, f"sample_no_fire_{i:02d}.npy")
            np.save(filename, patch)
            
            print(f"  ✓ Saved {filename} (no fire)")
            saved_count += 1
    
    print(f"\n[extract] Extracted {saved_count} samples to {output_dir}/")
    print(f"\nYou can now use these samples to test the webapp:")
    print(f"  1. Open http://127.0.0.1:5000")
    print(f"  2. Click 'Choose File' and select any .npy file from {output_dir}/")
    print(f"  3. Click 'Predict Fire Spread' to get predictions")


if __name__ == "__main__":
    extract_samples(output_dir="data", num_samples=10)
