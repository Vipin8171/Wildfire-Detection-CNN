"""
Training Script for Wildfire Spread Segmentation
=================================================
Trains Attention U-Net on the Next Day Wildfire Spread dataset.
Supports mixed precision, learning-rate scheduling, early stopping,
checkpoint saving, and TensorBoard logging.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_dataloaders
from src.models import get_model


# ──────────────────── Metrics ────────────────────
def compute_metrics(logits, targets, weights, threshold=0.5):
    """Compute segmentation metrics on valid (known) pixels only."""
    probs = torch.sigmoid(logits.squeeze(1))      # (B, H, W)
    preds = (probs >= threshold).float()

    mask = weights > 0
    preds_m = preds[mask]
    targets_m = targets[mask]

    tp = ((preds_m == 1) & (targets_m == 1)).sum().float()
    fp = ((preds_m == 1) & (targets_m == 0)).sum().float()
    fn = ((preds_m == 0) & (targets_m == 1)).sum().float()
    tn = ((preds_m == 0) & (targets_m == 0)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "tn": tn.item(),
    }


# ──────────────────── Training loop ────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    all_metrics = []

    for batch_idx, (x, label, weight) in enumerate(loader):
        x = x.to(device)
        label = label.to(device)
        weight = weight.to(device)

        optimizer.zero_grad()
        use_amp = device.type == "cuda"
        with autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, label, weight)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            metrics = compute_metrics(logits, label, weight)
            all_metrics.append(metrics)

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = []

    for x, label, weight in loader:
        x = x.to(device)
        label = label.to(device)
        weight = weight.to(device)

        logits = model(x)
        loss = criterion(logits, label, weight)
        total_loss += loss.item()
        metrics = compute_metrics(logits, label, weight)
        all_metrics.append(metrics)

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return avg_loss, avg_metrics


# ──────────────────── Main ────────────────────
def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    print("[train] Loading data …")
    train_loader, val_loader, test_loader, data_info = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        train_shards=args.train_shards,
    )

    # Save data info for inference
    info_path = os.path.join(args.save_dir, "data_info.json")
    data_info_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v
                              for k, v in data_info.items()}
    with open(info_path, "w") as f:
        json.dump(data_info_serializable, f, indent=2)
    print(f"[train] Data info saved to {info_path}")

    # Model
    model, criterion = get_model(
        args.model_name,
        in_channels=12,
        pos_weight=data_info["pos_weight"],
    )
    model = model.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # Training
    best_val_iou = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [],
               "train_f1": [], "val_f1": [], "lr": []}

    print(f"\n{'='*60}")
    print(f"[train] Starting training for {args.epochs} epochs")
    print(f"[train] Model: {args.model_name}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["lr"].append(lr)

        print(f"Epoch {epoch:03d}/{args.epochs} ({elapsed:.1f}s)  "
              f"Train Loss: {train_loss:.4f} IoU: {train_metrics['iou']:.4f} F1: {train_metrics['f1']:.4f}  │  "
              f"Val Loss: {val_loss:.4f} IoU: {val_metrics['iou']:.4f} F1: {val_metrics['f1']:.4f}  "
              f"LR: {lr:.6f}")

        # Save best
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            patience_counter = 0
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": best_val_iou,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  ★ New best model saved (IoU: {best_val_iou:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[train] Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Save final model & history
    final_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[train] Training complete!")
    print(f"  Best val IoU: {best_val_iou:.4f}")
    print(f"  Models saved in: {args.save_dir}")
    print(f"  History saved to: {history_path}")

    return model, history, data_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wildfire Segmentation Model")
    parser.add_argument("--model_name", type=str, default="unet",
                        choices=["unet", "unet_lite"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_shards", type=int, default=None,
                        help="Limit training shards (default: all)")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "checkpoints"))
    args = parser.parse_args()
    main(args)
