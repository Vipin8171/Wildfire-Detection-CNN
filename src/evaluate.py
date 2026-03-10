"""
Evaluation Script for Wildfire Spread Segmentation
===================================================
Loads a trained model checkpoint, runs on the test set,
computes metrics, generates visualisations, and saves reports.
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
from sklearn.metrics import roc_curve, auc, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_dataloaders, INPUT_FEATURES, FEATURE_INFO
from src.models import get_model
from src.train import compute_metrics


# ──────────────────── helpers ────────────────────
def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[eval] Loaded checkpoint from {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    return model


# ──────────────────── Full evaluation ────────────────────
@torch.no_grad()
def evaluate_model(model, loader, criterion, device, threshold=0.5):
    model.eval()
    all_logits, all_targets, all_weights = [], [], []
    total_loss = 0

    for x, label, weight in loader:
        x = x.to(device)
        label = label.to(device)
        weight = weight.to(device)
        logits = model(x)
        loss = criterion(logits, label, weight)
        total_loss += loss.item()

        all_logits.append(logits.cpu())
        all_targets.append(label.cpu())
        all_weights.append(weight.cpu())

    logits_cat = torch.cat(all_logits)       # (N, 1, 64, 64)
    targets_cat = torch.cat(all_targets)     # (N, 64, 64)
    weights_cat = torch.cat(all_weights)     # (N, 64, 64)
    avg_loss = total_loss / len(loader)

    # Overall metrics
    metrics = compute_metrics(logits_cat, targets_cat, weights_cat, threshold)
    metrics["loss"] = avg_loss

    # For ROC / PR curves
    probs = torch.sigmoid(logits_cat.squeeze(1))
    mask = weights_cat > 0
    y_true = targets_cat[mask].numpy()
    y_score = probs[mask].numpy()

    return metrics, logits_cat, targets_cat, weights_cat, y_true, y_score


# ──────────────────── Visualisations ────────────────────
def plot_training_history(history_path, save_dir):
    with open(history_path) as f:
        h = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(h["train_loss"], label="Train Loss")
    axes[0].plot(h["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IoU
    axes[1].plot(h["train_iou"], label="Train IoU")
    axes[1].plot(h["val_iou"], label="Val IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Intersection over Union")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LR
    axes[2].plot(h["lr"], label="Learning Rate", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[eval] Training curves saved")


def plot_roc_pr(y_true, y_score, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, "b-", label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, "r-", label=f"PR (AUC = {pr_auc:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_pr_curves.png"), dpi=150)
    plt.close()
    print(f"[eval] ROC & PR curves saved  (ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f})")
    return roc_auc, pr_auc


def plot_sample_predictions(logits, targets, weights, save_dir, n=8, threshold=0.5):
    """Visualise predictions on sample patches."""
    probs = torch.sigmoid(logits.squeeze(1)).numpy()
    preds = (probs >= threshold).astype(float)
    targets_np = targets.numpy()
    weights_np = weights.numpy()

    # Pick samples with some fire pixels
    fire_counts = (targets_np == 1).sum(axis=(1, 2))
    top_idx = np.argsort(fire_counts)[-n:]

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    cmap_fire = mcolors.ListedColormap(["black", "red"])
    cmap_prob = "hot"

    for row, idx in enumerate(top_idx):
        # Ground truth
        axes[row, 0].imshow(targets_np[idx], cmap=cmap_fire, vmin=0, vmax=1)
        axes[row, 0].set_title(f"Ground Truth (fire px: {int(fire_counts[idx])})")

        # Probability map
        axes[row, 1].imshow(probs[idx], cmap=cmap_prob, vmin=0, vmax=1)
        axes[row, 1].set_title("Predicted Probability")

        # Binary prediction
        axes[row, 2].imshow(preds[idx], cmap=cmap_fire, vmin=0, vmax=1)
        axes[row, 2].set_title("Binary Prediction")

        # Weight mask (known vs unknown)
        axes[row, 3].imshow(weights_np[idx], cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title("Weight Mask (white=known)")

        for col in range(4):
            axes[row, col].axis("off")

    plt.suptitle("Sample Predictions (top fire-pixel patches)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sample_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Sample predictions saved ({n} samples)")


def plot_feature_importance(model, loader, device, save_dir, n_batches=10):
    """Simple occlusion-based feature importance: zero out each channel and measure IoU drop."""
    model.eval()
    # Get baseline IoU
    base_metrics_list = []
    batches = []
    for i, (x, label, weight) in enumerate(loader):
        if i >= n_batches:
            break
        batches.append((x, label, weight))
        x = x.to(device)
        logits = model(x)
        m = compute_metrics(logits, label.to(device), weight.to(device))
        base_metrics_list.append(m["iou"])
    base_iou = np.mean(base_metrics_list)

    importance = {}
    for ch_idx, feat_name in enumerate(INPUT_FEATURES):
        drop_ious = []
        for x, label, weight in batches:
            x_mod = x.clone()
            x_mod[:, ch_idx] = 0  # Zero out this channel
            x_mod = x_mod.to(device)
            logits = model(x_mod)
            m = compute_metrics(logits, label.to(device), weight.to(device))
            drop_ious.append(m["iou"])
        importance[feat_name] = base_iou - np.mean(drop_ious)

    # Plot
    names = list(importance.keys())
    values = list(importance.values())
    sorted_idx = np.argsort(values)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([names[i] for i in sorted_idx], [values[i] for i in sorted_idx], color="coral")
    ax.set_xlabel("IoU Drop (higher = more important)")
    ax.set_title("Feature Importance (Occlusion-based)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"[eval] Feature importance saved")
    return importance


# ──────────────────── Main ────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="unet_lite")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_dir, "checkpoints")
    if args.checkpoint is None:
        args.checkpoint = os.path.join(ckpt_dir, "best_model.pth")
    if args.save_dir is None:
        args.save_dir = os.path.join(project_dir, "results")
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # Load data info
    info_path = os.path.join(ckpt_dir, "data_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            data_info = json.load(f)
        pos_weight = data_info.get("pos_weight", 20.0)
    else:
        pos_weight = 20.0

    # Model
    model, criterion = get_model(args.model_name, in_channels=12, pos_weight=pos_weight)
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # Data
    _, _, test_loader, _ = get_dataloaders(batch_size=args.batch_size)

    # Evaluate
    print("\n[eval] Evaluating on test set …")
    metrics, logits, targets, weights, y_true, y_score = evaluate_model(
        model, test_loader, criterion, device
    )

    print(f"\n{'='*50}")
    print("       TEST SET RESULTS")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")
    print(f"{'='*50}\n")

    # Save metrics
    metrics_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    history_path = os.path.join(ckpt_dir, "training_history.json")
    if os.path.exists(history_path):
        plot_training_history(history_path, args.save_dir)

    roc_auc, pr_auc = plot_roc_pr(y_true, y_score, args.save_dir)
    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc

    plot_sample_predictions(logits, targets, weights, args.save_dir)

    # Feature importance
    _, _, test_loader2, _ = get_dataloaders(batch_size=16)
    importance = plot_feature_importance(model, test_loader2, device, args.save_dir)
    metrics["feature_importance"] = importance

    # Final save
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[eval] All results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
