from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_default_transforms(img_size: int = 64):
    """
    Returns train (augmented) and eval transforms for EuroSAT RGB images.
    Train gets augmentation; val/test stay deterministic.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),

        # Augmentations (train only)
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
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    return train_transform, eval_transform


def load_eurosat_dataset(
    data_dir: Path,
    img_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Loads the EuroSAT RGB dataset from a directory structured like:
        data_dir/
            AnnualCrop/
            Forest/
            ...
            SeaLake/

    Returns train, val, test dataloaders and class names.
    """

    data_dir = Path(data_dir)
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    train_transform, eval_transform = get_default_transforms(img_size)

    # ImageFolder expects subfolders by class
    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    class_names = full_dataset.classes

    # Split sizes
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # IMPORTANT:
    # random_split returns Subset objects that reference the SAME underlying dataset.
    # If we set val_dataset.dataset.transform, it changes it for train too.
    # So we create separate datasets for val/test using the same root but eval_transform.

    val_dataset_full = datasets.ImageFolder(root=str(data_dir), transform=eval_transform)
    test_dataset_full = datasets.ImageFolder(root=str(data_dir), transform=eval_transform)

    # Rebuild subsets with same indices
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_dataset.indices)
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_dataset.indices)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
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