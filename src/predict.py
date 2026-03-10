"""
Prediction / Inference Script
==============================
Run the trained model on new satellite patches or the test set,
visualise results, and export predictions.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    load_tfrecord_data, normalize, compute_channel_stats,
    INPUT_FEATURES, FEATURE_INFO, PATCH_SIZE, NUM_CHANNELS,
)
from src.models import get_model


def load_model(checkpoint_path, model_name="unet", device="cpu"):
    """Load trained model from checkpoint."""
    model, _ = get_model(model_name, in_channels=NUM_CHANNELS)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[predict] Model loaded from {checkpoint_path}")
    return model


def predict_patch(model, patch, means, stds, device="cpu", threshold=0.5):
    """
    Predict fire mask for a single 64×64 patch.

    Parameters
    ----------
    patch : np.ndarray (12, 64, 64) — raw satellite features
    means, stds : np.ndarray (12,) — normalization stats

    Returns
    -------
    prob_map : np.ndarray (64, 64) — fire probability [0, 1]
    pred_mask : np.ndarray (64, 64) — binary fire prediction
    """
    # Normalize
    patch_norm = (patch - means[:, None, None]) / np.where(stds == 0, 1, stds)[:, None, None]
    patch_norm = np.nan_to_num(patch_norm, nan=0.0)

    # To tensor
    x = torch.from_numpy(patch_norm).unsqueeze(0).float().to(device)  # (1, 12, 64, 64)

    with torch.no_grad():
        logits = model(x)                                              # (1, 1, 64, 64)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()           # (64, 64)

    pred = (prob >= threshold).astype(np.float32)
    return prob, pred


def predict_batch(model, patches, means, stds, device="cpu", threshold=0.5):
    """Predict on a batch of patches. patches: (N, 12, 64, 64)."""
    stds_safe = np.where(stds == 0, 1, stds)
    patches_norm = (patches - means[None, :, None, None]) / stds_safe[None, :, None, None]
    patches_norm = np.nan_to_num(patches_norm, nan=0.0)

    x = torch.from_numpy(patches_norm).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    preds = (probs >= threshold).astype(np.float32)
    return probs, preds


def visualize_prediction(patch, prob_map, pred_mask, ground_truth=None, save_path=None):
    """
    Visualise a single prediction with feature channels.

    Shows: NDVI, elevation, temperature, prev fire → probability → prediction (+ GT if available).
    """
    n_cols = 6 if ground_truth is not None else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    cmap_fire = mcolors.ListedColormap(["black", "red"])

    # Key features
    feature_plots = [
        (0, "NDVI", "RdYlGn"),
        (1, "Elevation", "terrain"),
        (5, "Max Temp", "hot"),
        (11, "Prev Fire", cmap_fire),
    ]
    for col, (ch_idx, title, cmap) in enumerate(feature_plots):
        im = axes[col].imshow(patch[ch_idx], cmap=cmap)
        axes[col].set_title(title, fontsize=10)
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    # Probability
    im = axes[4].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[4].set_title("Fire Probability", fontsize=10)
    axes[4].axis("off")
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    if ground_truth is not None:
        axes[5].imshow(ground_truth, cmap=cmap_fire, vmin=0, vmax=1)
        axes[5].set_title("Ground Truth", fontsize=10)
        axes[5].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[predict] Saved visualisation: {save_path}")
    plt.close()


def generate_prediction_report(patch, prob_map, pred_mask, feature_means=None):
    """Generate a text report about the prediction."""
    fire_pixels = pred_mask.sum()
    total_pixels = pred_mask.size
    fire_percentage = (fire_pixels / total_pixels) * 100

    report = {
        "fire_detected": bool(fire_pixels > 0),
        "fire_pixels": int(fire_pixels),
        "total_pixels": int(total_pixels),
        "fire_percentage": round(fire_percentage, 2),
        "max_probability": round(float(prob_map.max()), 4),
        "mean_probability": round(float(prob_map.mean()), 4),
        "risk_level": "HIGH" if fire_percentage > 5 else "MEDIUM" if fire_percentage > 1 else "LOW",
    }

    # Feature summaries
    feature_summary = {}
    for i, feat_name in enumerate(INPUT_FEATURES):
        ch = patch[i]
        valid = ch[~np.isnan(ch)]
        if len(valid) > 0:
            feature_summary[feat_name] = {
                "full_name": FEATURE_INFO[feat_name]["full_name"],
                "mean": round(float(valid.mean()), 4),
                "min": round(float(valid.min()), 4),
                "max": round(float(valid.max()), 4),
                "source": FEATURE_INFO[feat_name]["source"],
                "unit": FEATURE_INFO[feat_name]["unit"],
            }
    report["feature_summary"] = feature_summary
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="unet")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.checkpoint is None:
        args.checkpoint = os.path.join(project_dir, "checkpoints", "best_model.pth")
    if args.save_dir is None:
        args.save_dir = os.path.join(project_dir, "results", "predictions")
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data info
    info_path = os.path.join(project_dir, "checkpoints", "data_info.json")
    with open(info_path) as f:
        data_info = json.load(f)
    means = np.array(data_info["means"], dtype=np.float32)
    stds = np.array(data_info["stds"], dtype=np.float32)

    # Load model
    model = load_model(args.checkpoint, args.model_name, device)

    # Load data
    X, y = load_tfrecord_data(args.split)

    # Pick samples with most fire pixels
    fire_counts = (y == 1).sum(axis=(1, 2))
    top_idx = np.argsort(fire_counts)[-args.n_samples:]

    all_reports = []
    for i, idx in enumerate(top_idx):
        patch = X[idx]  # (12, 64, 64)
        gt = (y[idx] == 1).astype(np.float32)

        prob, pred = predict_patch(model, patch, means, stds, device, args.threshold)

        # Visualise
        save_path = os.path.join(args.save_dir, f"prediction_{i:03d}.png")
        visualize_prediction(patch, prob, pred, ground_truth=gt, save_path=save_path)

        # Report
        report = generate_prediction_report(patch, prob, pred)
        report["sample_index"] = int(idx)
        report["ground_truth_fire_pixels"] = int(gt.sum())
        all_reports.append(report)

        print(f"  Sample {i}: fire={report['fire_detected']}, "
              f"pred_fire_px={report['fire_pixels']}, "
              f"gt_fire_px={report['ground_truth_fire_pixels']}, "
              f"risk={report['risk_level']}")

    # Save reports
    report_path = os.path.join(args.save_dir, "prediction_reports.json")
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n[predict] {len(all_reports)} prediction reports saved to {report_path}")


if __name__ == "__main__":
    main()
