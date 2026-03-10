"""
Dataset Download Script
========================
Downloads the Next Day Wildfire Spread dataset using kagglehub.
"""

import os

def download_dataset():
    try:
        import kagglehub
        path = kagglehub.dataset_download("fantineh/next-day-wildfire-spread")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nManual download:")
        print("  1. pip install kagglehub")
        print("  2. python -c \"import kagglehub; kagglehub.dataset_download('fantineh/next-day-wildfire-spread')\"")
        print("  3. Or download from: https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread")
        return None

if __name__ == "__main__":
    download_dataset()
