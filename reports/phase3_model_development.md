# Phase 3: Model Development

## Architecture: Attention U-Net

### Why U-Net?
- Standard architecture for **semantic segmentation** tasks
- Skip connections preserve spatial detail (critical for 64×64 fire masks)
- Proven in satellite/remote-sensing applications

### Enhancements
1. **Attention Gates** (Oktay et al., 2018): Focus on fire-relevant spatial regions in skip connections
2. **Squeeze-and-Excitation Blocks**: Channel attention in bottleneck — learns which satellite features are most informative
3. **Masked Loss**: Unknown pixels (label=-1) receive zero weight, preventing noise in training
4. **BCE + Dice Loss**: Combined loss handles pixel imbalance — BCE with positive weighting + Dice for spatial overlap

### Architecture Details
| Stage | Channels | Spatial Size |
|-------|----------|-------------|
| Input | 12 | 64×64 |
| Encoder 1 | 64 | 64×64 |
| Encoder 2 | 128 | 32×32 |
| Encoder 3 | 256 | 16×16 |
| Encoder 4 | 512 | 8×8 |
| Bottleneck | 512 + SE | 4×4 |
| Decoder 4 | 512 | 8×8 |
| Decoder 3 | 256 | 16×16 |
| Decoder 2 | 128 | 32×32 |
| Decoder 1 | 64 | 64×64 |
| Output | 1 (sigmoid) | 64×64 |

### Parameters
- Full U-Net: ~31M parameters
- Lite U-Net: ~8M parameters
