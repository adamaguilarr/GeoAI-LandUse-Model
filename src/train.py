from pathlib import Path
import random
from typing import Optional, Tuple, List, Dict, Any

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

    # Optional: more reproducible, slightly slower (mostly matters on GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    data_dir,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    img_size: int = 64,
    device: Optional[str] = None,
    save_best: bool = True,
    seed: int = 42,
    aug_level: str = "light",
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    early_stop_patience: int = 4,          # stop if no val improvement for N epochs
    min_delta: float = 1e-4,               # required improvement to reset patience
) -> Tuple[torch.nn.Module, List[str]]:
    seed_everything(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Stable splits + augment control live in dataset.py
    train_loader, val_loader, test_loader, class_names = load_eurosat_dataset(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        aug_level=aug_level,
    )

    model = build_model(num_classes=len(class_names)).to(device)

    # Label smoothing is a small but often real win
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save both "best" and "last" so you always have something to inspect
    best_path = models_dir / "simple_cnn_v2_best.pth"
    last_path = models_dir / "simple_cnn_v2_last.pth"
    meta_path = best_path.with_suffix(".meta.pt")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    def get_lr(opt):
        return opt.param_groups[0]["lr"]

    best_val_acc = -1.0
    epochs_no_improve = 0
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

        # Always save "last"
        torch.save(model.state_dict(), last_path)

        # Scheduler step on val metric
        scheduler.step(val_acc)

        new_lr = get_lr(optimizer)
        if new_lr != current_lr:
            current_lr = new_lr
            print(f"LR changed -> {current_lr:.2e}")

        # -------- Save best / early stopping --------
        improved = (val_acc - best_val_acc) > min_delta

        if save_best and improved:
            best_val_acc = val_acc
            epochs_no_improve = 0

            torch.save(model.state_dict(), best_path)
            torch.save(
                {
                    "class_names": class_names,
                    "img_size": img_size,
                    "batch_size": batch_size,
                    "seed": seed,
                    "aug_level": aug_level,
                    "epochs_trained": epoch + 1,
                    "lr_init": lr,
                    "weight_decay": weight_decay,
                    "label_smoothing": label_smoothing,
                    "best_val_acc": float(best_val_acc),
                    "best_path": str(best_path),
                    "last_path": str(last_path),
                },
                meta_path,
            )
            print(f"Saved BEST model so far to {best_path} (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1

        if save_best and early_stop_patience is not None and epochs_no_improve >= early_stop_patience:
            print(f"Early stopping: no val improvement for {early_stop_patience} epoch(s).")
            break

    # Load best weights back into the returned model
    if save_best and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

    print(f"Done. Best Val Acc: {best_val_acc:.4f}")
    return model, class_names