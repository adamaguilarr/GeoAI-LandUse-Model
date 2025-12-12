from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.dataset import load_eurosat_dataset
from src.models import build_model

# repo root: .../GeoAI-LandUse-Model
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_model(
    data_dir,
    epochs=5,
    lr=1e-3,
    batch_size=64,
    img_size=64,
    device=None,
    save_best=True,  # save best val-acc model
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = load_eurosat_dataset(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    save_path = PROJECT_ROOT / "models" / "simple_cnn.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0

    for epoch in range(epochs):
        # -------- Training --------
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        # -------- Validation --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(1, total)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # -------- Save model --------
        if save_best:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"Saved BEST model so far to {save_path} (val_acc={best_val_acc:.4f})")
        else:
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    print(f"Done. Best Val Acc: {best_val_acc:.4f}" if save_best else "Done.")
    return model, class_names
