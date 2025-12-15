from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.dataset import load_eurosat_dataset
from src.models import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional: more reproducible, slightly slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    data_dir,
    epochs=5,
    lr=1e-3,
    batch_size=64,
    img_size=64,
    device=None,
    save_best=True,
    seed=42,              # NEW: lock dataset split + reproducibility
    aug_level="light",    # keep your aug switch here
):
    seed_everything(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # IMPORTANT: pass seed so train/val/test split is stable
    train_loader, val_loader, test_loader, class_names = load_eurosat_dataset(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        aug_level=aug_level,
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    save_path = PROJECT_ROOT / "models" / "simple_cnn_v2_best.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: also save a tiny metadata file to avoid confusion later
    meta_path = save_path.with_suffix(".meta.pt")

    best_val_acc = -1.0

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    def get_lr(opt):
        return opt.param_groups[0]["lr"]

    current_lr = get_lr(optimizer)

    for epoch in range(epochs):
        # -------- Training --------
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
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
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        scheduler.step(val_acc)

        new_lr = get_lr(optimizer)
        if new_lr != current_lr:
            current_lr = new_lr
            print(f"LR changed -> {current_lr:.2e}")

        # -------- Save model --------
        if save_best:
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                torch.save(model.state_dict(), save_path)
                torch.save(
                    {
                        "class_names": class_names,
                        "img_size": img_size,
                        "batch_size": batch_size,
                        "seed": seed,
                        "aug_level": aug_level,
                        "best_val_acc": best_val_acc,
                    },
                    meta_path
                )

                print(f"Saved BEST model so far to {save_path} (val_acc={best_val_acc:.4f})")
        else:
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    # NEW: load best weights back into the model before returning
    if save_best and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()

    print(f"Done. Best Val Acc: {best_val_acc:.4f}")
    return model, class_names