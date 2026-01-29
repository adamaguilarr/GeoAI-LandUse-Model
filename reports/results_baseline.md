# Model Results Summary

This document summarizes the baseline CNN results and transfer learning experiments conducted for the GeoAI Land Use Model project using the EuroSAT dataset.

---

## Baseline Results — SimpleCNN v2

### Dataset
- EuroSAT RGB (ImageFolder)
- Image size: 64 × 64
- Split: train / validation / test (seed = 42)

### Training Configuration
- Model: SimpleCNN v2
- Epochs: 15 (early stopping patience = 4)
- Optimizer: Adam
- Learning rate: 1e-3
- Weight decay: 1e-4
- Label smoothing: 0.05
- Scheduler: ReduceLROnPlateau (factor = 0.5, patience = 2)
- Augmentation level: light

### Best Validation
- Best validation accuracy: **0.9440**
- Epoch: 13 / 15

### Test Performance
- Test accuracy: **0.944**
- Macro average F1-score: **0.9432**
- Weighted average F1-score: **0.9441**

### Notes
- Strong performance across most land-use classes
- Primary confusions occur between visually similar categories
- Confusion matrices saved to:
  - `reports/figures/confusion_matrix_v2_normalized.png`
  - `reports/figures/confusion_matrix_v2_raw.png`

---

## Transfer Learning Results — ResNet18 (Pretrained, Fine-Tuned)

### Dataset
- EuroSAT RGB (ImageFolder)
- Image size: 64 × 64
- Split: train / validation / test (seed = 42)

### Training Configuration
- Model: ResNet18 (ImageNet pretrained)
- Phase 1: Frozen backbone (train classification head only)
- Phase 2: Full fine-tuning (all layers unfrozen)
- Epochs (fine-tuning): 10
- Optimizer: Adam
- Learning rate: 1e-4
- Weight decay: 1e-4
- Label smoothing: 0.05
- Augmentation level: light
- Early stopping patience: 3

### Best Validation
- Best validation accuracy: **0.9738**

### Test Performance
- Test accuracy: **0.9721**
- Macro average F1-score: **0.9716**
- Weighted average F1-score: **0.9721**

### Notes
- Fine-tuning significantly outperforms both the frozen ResNet and the CNN baseline
- Improvements observed across all land-use classes
- Remaining errors largely reflect semantic ambiguity at 64×64 resolution
- Confusion matrices saved to:
  - `reports/figures/confusion_matrix_resnet18_ft_normalized.png`
  - `reports/figures/confusion_matrix_resnet18_ft_raw.png`

---

## Model Comparison Summary

| Model                     | Test Accuracy |
|---------------------------|---------------|
| SimpleCNN v2              | ~0.94         |
| ResNet18 (Frozen)         | ~0.80         |
| ResNet18 (Fine-tuned)     | **~0.97**     |

---

## Takeaways
- A well-regularized CNN baseline achieves strong performance on EuroSAT
- Transfer learning with ResNet18 provides a substantial accuracy gain
- Most remaining errors are between semantically similar land-use classes
- The workflow is fully reproducible and ready for GeoAI extensions