# Baseline Results (SimpleCNN v2)

## Dataset
- EuroSAT RGB (ImageFolder)
- img_size: 64
- split: train/val/test with seed=42

## Training config
- model: SimpleCNN v2
- epochs: 15 (early stopping patience=4)
- optimizer: Adam
- lr: 1e-3
- weight_decay: 1e-4
- label_smoothing: 0.05
- scheduler: ReduceLROnPlateau(factor=0.5, patience=2)
- aug_level: light

## Best validation
- best_val_acc: 0.9440
- epoch: 13/15

## Test performance
- test_acc: 0.944
- macro avg f1: 0.9432
- weighted avg f1: 0.9441

## Notes
- Confusion matrices saved to:
  - reports/figures/confusion_matrix_v2_normalized.png
  - reports/figures/confusion_matrix_v2_raw.png