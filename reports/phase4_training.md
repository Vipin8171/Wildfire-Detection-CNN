# Phase 4: Training & Evaluation

## Training Configuration
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: Cosine annealing to lr×0.01
- **Augmentation**: Random horizontal/vertical flips + 90° rotations
- **Batch size**: 32
- **Epochs**: Up to 50 (early stopping patience=10)
- **Mixed precision**: AMP on CUDA

## Metrics
- **IoU** (Intersection over Union): Primary metric for fire mask overlap
- **Dice/F1 Score**: Harmonic mean of precision and recall
- **Precision**: Of predicted fire pixels, how many are correct
- **Recall**: Of actual fire pixels, how many are detected
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve

## Results
*(Updated after training)*

| Metric | Value |
|--------|-------|
| Test Loss | — |
| Test IoU | — |
| Test F1 | — |
| Test Precision | — |
| Test Recall | — |
| ROC-AUC | — |
| PR-AUC | — |
