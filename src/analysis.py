from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: str,
) -> Tuple[List[int], List[int], List[torch.Tensor]]:
    """
    Returns:
      y_true: list of true labels
      y_pred: list of predicted labels
      images: list of image tensors (CHW) from the loader (unnormalized not guaranteed)
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    imgs: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        # store CPU tensors for saving later
        imgs.extend(images.cpu())

    return y_true, y_pred, imgs


def print_worst_classes_by_recall(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    top_k: int = 5,
) -> None:
    """
    Recall per class = TP / (TP + FN)
    """
    num_classes = len(class_names)
    tp = [0] * num_classes
    fn = [0] * num_classes

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1

    recall = []
    for i in range(num_classes):
        denom = tp[i] + fn[i]
        r = tp[i] / denom if denom > 0 else 0.0
        recall.append((r, i, tp[i], fn[i]))

    recall.sort(key=lambda x: x[0])  # lowest recall first

    print(f"\nWorst {top_k} classes by recall:")
    for r, i, tp_i, fn_i in recall[:top_k]:
        print(f"- {class_names[i]:<22} recall={r:.3f} (TP={tp_i}, FN={fn_i})")


def _denormalize_image(img: torch.Tensor, mean, std) -> torch.Tensor:
    """
    img: CHW tensor normalized
    returns CHW tensor in [0,1] approx
    """
    out = img.clone()
    for c in range(out.shape[0]):
        out[c] = out[c] * std[c] + mean[c]
    return out.clamp(0, 1)


def save_misclassified_images(
    y_true: List[int],
    y_pred: List[int],
    images: List[torch.Tensor],
    class_names: List[str],
    out_dir: Path,
    max_per_pair: int = 20,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> None:
    """
    Saves misclassified images into out_dir with filenames that encode true/pred.
    Limits saves per (true,pred) confusion pair.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[Tuple[int, int], int] = {}

    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue

        key = (t, p)
        counts[key] = counts.get(key, 0)
        if counts[key] >= max_per_pair:
            continue

        img = images[idx]
        img = _denormalize_image(img, mean, std)
        pil = to_pil_image(img)

        true_name = class_names[t]
        pred_name = class_names[p]

        filename = f"true_{true_name}_pred_{pred_name}_{counts[key]:03d}.png"
        pil.save(out_dir / filename)

        counts[key] += 1

    print(f"Saved misclassified images to: {out_dir}")