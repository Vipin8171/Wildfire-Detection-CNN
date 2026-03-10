"""
Models for Wildfire Spread Segmentation
========================================
U-Net architecture with attention gates, designed for 12-channel
satellite input (64×64 patches) → binary fire mask output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────── Building blocks ────────────────────
class ConvBlock(nn.Module):
    """Two 3×3 convolutions + BatchNorm + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections (Oktay et al., 2018)."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        # Ensure same spatial size
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        att = self.relu(g + s)
        att = self.psi(att)
        return skip * att


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ──────────────────── U-Net with Attention ────────────────────
class WildfireUNet(nn.Module):
    """
    Attention U-Net for wildfire spread prediction.

    Input:  (B, 12, 64, 64)  — 12-channel satellite features
    Output: (B, 1, 64, 64)   — fire probability map (sigmoid)

    Architecture:
        Encoder: 4 down-sampling stages (12→64→128→256→512)
        Bottleneck: 512 channels with SE block
        Decoder: 4 up-sampling stages with attention gates
        Head: 1×1 conv → sigmoid
    """

    def __init__(self, in_channels=12, base_filters=64):
        super().__init__()
        f = base_filters  # 64

        # ── Encoder ──
        self.enc1 = ConvBlock(in_channels, f)        # 64×64 → 64×64,  f
        self.enc2 = ConvBlock(f, f * 2)               # 32×32,  2f
        self.enc3 = ConvBlock(f * 2, f * 4)            # 16×16,  4f
        self.enc4 = ConvBlock(f * 4, f * 8)            #  8×8,   8f

        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ──
        self.bottleneck = nn.Sequential(
            ConvBlock(f * 8, f * 8),
            SEBlock(f * 8),
        )

        # ── Attention gates ──
        self.att4 = AttentionGate(f * 8, f * 8, f * 4)
        self.att3 = AttentionGate(f * 4, f * 4, f * 2)
        self.att2 = AttentionGate(f * 2, f * 2, f)
        self.att1 = AttentionGate(f,     f,     f // 2)

        # ── Decoder ──
        self.up4 = nn.ConvTranspose2d(f * 8, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        # ── Head ──
        self.head = nn.Conv2d(f, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                           # (B, 64,  64, 64)
        e2 = self.enc2(self.pool(e1))                # (B, 128, 32, 32)
        e3 = self.enc3(self.pool(e2))                # (B, 256, 16, 16)
        e4 = self.enc4(self.pool(e3))                # (B, 512,  8,  8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))           # (B, 512,  4,  4)

        # Decoder
        d4 = self.up4(b)                             # (B, 512,  8,  8)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # (B, 512,  8,  8)

        d3 = self.up3(d4)                            # (B, 256, 16, 16)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # (B, 256, 16, 16)

        d2 = self.up2(d3)                            # (B, 128, 32, 32)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 128, 32, 32)

        d1 = self.up1(d2)                            # (B,  64, 64, 64)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B,  64, 64, 64)

        return self.head(d1)                         # (B,   1, 64, 64)  logits


# ──────────────────── Lightweight variant ────────────────────
class WildfireUNetLite(nn.Module):
    """Smaller U-Net (base_filters=32) for faster training / prototyping."""

    def __init__(self, in_channels=12):
        super().__init__()
        self.net = WildfireUNet(in_channels=in_channels, base_filters=32)

    def forward(self, x):
        return self.net(x)


# ──────────────────── Loss functions ────────────────────
class MaskedBCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss with pixel-weight masking.
    Unknown pixels (weight=0) are excluded from loss computation.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight  # scalar float

    def forward(self, logits, targets, weights):
        """
        logits  : (B, 1, H, W)  raw logits
        targets : (B, H, W)     0/1
        weights : (B, H, W)     0 for unknown, 1 for known
        """
        logits = logits.squeeze(1)             # (B, H, W)
        probs = torch.sigmoid(logits)

        # ── Masked BCE ──
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, weight=weights, pos_weight=pw, reduction="sum"
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, weight=weights, reduction="sum"
            )
        num_valid = weights.sum().clamp(min=1)
        bce = bce / num_valid

        # ── Masked Dice ──
        probs_masked = probs * weights
        targets_masked = targets * weights
        intersection = (probs_masked * targets_masked).sum()
        dice = 1 - (2 * intersection + 1) / (probs_masked.sum() + targets_masked.sum() + 1)

        return self.bce_weight * bce + self.dice_weight * dice


# ──────────────────── Utility ────────────────────
def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model(model_name="unet", in_channels=12, pos_weight=None):
    """Factory function to create model + loss."""
    if model_name == "unet":
        model = WildfireUNet(in_channels=in_channels)
    elif model_name == "unet_lite":
        model = WildfireUNetLite(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    criterion = MaskedBCEDiceLoss(pos_weight=pos_weight)
    total, trainable = count_parameters(model)
    print(f"[model] {model_name}: {total:,} params ({trainable:,} trainable)")
    return model, criterion


if __name__ == "__main__":
    model, criterion = get_model("unet", pos_weight=20.0)
    x = torch.randn(2, 12, 64, 64)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    # Test loss
    targets = torch.randint(0, 2, (2, 64, 64)).float()
    weights = torch.ones(2, 64, 64)
    loss = criterion(out, targets, weights)
    print(f"Loss:   {loss.item():.4f}")
