# GeoAI Land Use Model

A machine learning project that classifies satellite imagery into land-use categories using the EuroSAT dataset. This project explores remote sensing, computer vision, and geospatial analysis, with the goal of building a clear and reproducible GeoAI workflow.

---

## Project Goals

- Build a land-use classifier using satellite image patches.  
- Compare a simple CNN baseline with transfer learning models such as ResNet.  
- Evaluate classification performance and visualize results.  
- Lay the groundwork for optional extensions like mapping predictions or deploying the model as an API.

---

## Dataset

**EuroSAT (RGB version)**  
A public dataset containing 27,000 satellite images (64×64px) across 10 land-use classes, including Residential, Forest, Industrial, River, and more.

Dataset source:  
https://github.com/phelber/eurosat

Classes:

- AnnualCrop  
- Forest  
- HerbaceousVegetation  
- Highway  
- Industrial  
- Pasture  
- PermanentCrop  
- Residential  
- River  
- SeaLake  

---

## Planned Features

### Phase 1 — Core ML Workflow
- Dataset exploration and visualization  
- Baseline CNN training  
- Model evaluation (accuracy, confusion matrix, sample predictions)

### Phase 2 — Model Improvements
- Transfer learning using pretrained ResNet  
- Data augmentation  
- Hyperparameter tuning  

### Phase 3 — GeoAI Extensions
- Visualizing predictions on a map  
- Generating land-use overlays  
- Exporting results to GeoJSON or shapefile format  
- Building a small inference API (FastAPI)

---

## Current Results

### Baseline CNN (SimpleCNN v2)
- Test Accuracy: **~0.94**
- Strong performance on Forest, Residential, Industrial, and SeaLake
- Primary confusions occur between visually similar classes:
  - PermanentCrop vs AnnualCrop  
  - HerbaceousVegetation vs Pasture  
  - Highway vs Industrial  

### Transfer Learning (ResNet18)
- Frozen backbone: ~0.80 test accuracy
- Fine-tuned backbone: **~0.97 test accuracy**
- Significant improvement across all classes after fine-tuning
- Remaining errors largely reflect semantic ambiguity at 64×64 resolution