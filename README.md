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
- Test Accuracy ≈ 0.94
- Biggest Confusions:
    - HerbaceousVegetation - Pasture
    - Highway - Industrial
    - PermanentCrop - AnnualCrop

The CNN achieves strong performance on the EuroSAT dataset, with particularly high accuracy on Forest, Residential, Industrial, and SeaLake classes. Most misclassifications occur between semantically similar land-use types, such as AnnualCrop vs PermanentCrop and Highway vs River, reflecting inherent ambiguity at the given spatial resolution.