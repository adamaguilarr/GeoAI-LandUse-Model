import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path

from src.dataset import load_eurosat_dataset
from src.models import build_model


def train_model(
    data_dir,
    epochs=5,
    lr=1e-3,
    batch_size=64,
    img_size=64,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = load_eurosat_dataset(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

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

        train_loss = running_loss / len(train_loader)

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

        val_acc = correct / total

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # -------- Save model --------
    save_path = Path("models") / "simple_cnn.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), str(save_path))
    print(f"Model saved to {save_path}")

    return model, class_names