from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(img_size: int = 64, aug_level: str = "light"):
    """
    Returns (train_transform, eval_transform).

    aug_level: "none" | "light" | "medium" | "heavy"
    """
    aug_level = (aug_level or "light").lower()

    # Always deterministic for val/test
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if aug_level == "none":
        train_transform = eval_transform

    elif aug_level == "light":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    elif aug_level == "medium":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05),
                shear=5
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    elif aug_level == "heavy":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=35),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.12, 0.12),
                scale=(0.90, 1.10),
                shear=10
            ),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.20,
                hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    else:
        raise ValueError(f"Unknown aug_level='{aug_level}'. Use: none/light/medium/heavy")

    return train_transform, eval_transform


def load_eurosat_dataset(
    data_dir: Path,
    img_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    aug_level: str = "light",
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Loads EuroSAT RGB dataset from:
        data_dir/
            AnnualCrop/
            Forest/
            ...
            SeaLake/

    Returns: train_loader, val_loader, test_loader, class_names
    """

    data_dir = Path(data_dir)
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    train_transform, eval_transform = get_transforms(img_size=img_size, aug_level=aug_level)

    # Full dataset used only for indices/splits (train transform for train)
    full_train_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    class_names = full_train_dataset.classes

    total_size = len(full_train_dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_train_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # IMPORTANT: val/test must use eval transforms.
    # Create separate datasets and reuse the same indices.
    full_eval_dataset = datasets.ImageFolder(root=str(data_dir), transform=eval_transform)

    val_dataset = torch.utils.data.Subset(full_eval_dataset, val_subset.indices)
    test_dataset = torch.utils.data.Subset(full_eval_dataset, test_subset.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, class_names