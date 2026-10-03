"""
transforms.py
B7: reusable train and eval transforms for MILK10k.

All augmentation parameters are chosen from the colour-safety audit in
A3.5: they must not change the image more than the class gap itself.
The eval transform is fully deterministic.
"""
import json
from pathlib import Path
import torch
import torchvision.transforms as T

from . import config

NORM_PATH = config.CONFIGS_DIR / "norm_stats.json"


def _load_norm():
    if NORM_PATH.exists():
        with open(NORM_PATH) as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]
    # Fallback: ImageNet-like defaults. Should not be used in practice.
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_eval_transform(size=(224, 224)):
    """
    Deterministic eval transform: resize + ToTensor + Normalize.
    No randomness. Safe for val and test.
    """
    mean, std = _load_norm()
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_train_transform(size=(224, 224)):
    """
    Train transform: mild, medically safe augmentations.

    Parameters chosen from the A3.5 audit:
      - flips: label-invariant for skin lesions (no canonical orientation)
      - rotation ±15 deg: small, keeps the lesion inside the frame
      - ColorJitter: brightness 0.1, contrast 0.1, saturation 0.05, hue 0.02
        (hue shift stays under ~7 degrees, well inside the class gap)
      - RandomResizedCrop: scale in (0.85, 1.0) so the lesion stays in frame
    """
    mean, std = _load_norm()
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.RandomResizedCrop(size=size, scale=(0.85, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# Pre-built instances, so downstream code can just import them.
train_transform = build_train_transform()
eval_transform = build_eval_transform()


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reproducibility check on eval_transform
    sample_id = config.METADATA_PATH and __import__("pandas").read_csv(
        config.METADATA_PATH)["isic_id"].iloc[0]
    path = config.image_path(sample_id)
    img = Image.open(path).convert("RGB")

    t1 = eval_transform(img)
    t2 = eval_transform(img)
    assert torch.equal(t1, t2), "eval_transform is not deterministic"
    print("eval_transform deterministic:", torch.equal(t1, t2))
    print("tensor shape:", tuple(t1.shape), "dtype:", t1.dtype)
    print("min:", float(t1.min()), "max:", float(t1.max()),
          "mean:", float(t1.mean()))

    # Build the augmentation gallery: 1 original + 7 augmented, 3 classes
    out_dir = config.REPORTS_DIR / "figures_milestone1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick one image from 3 classes: BCC (common), MEL (clinically critical), DF (rare)
    import pandas as pd
    df = pd.read_csv(config.METADATA_PATH)
    picks = {}
    for cls, gt_path in [("BCC", "BCC"), ("MEL", "MEL"), ("DF", "DF")]:
        gt = pd.read_csv(config.GT_PATH)
        if gt_path in gt.columns:
            lesion = gt[gt[gt_path] == 1]["lesion_id"].iloc[0]
            row = df[df["lesion_id"] == lesion].iloc[0]
            picks[cls] = row["isic_id"]

    fig, axes = plt.subplots(len(picks), 8, figsize=(16, 6))
    for r, (cls, isic) in enumerate(picks.items()):
        img = Image.open(config.image_path(isic)).convert("RGB")
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{cls} - original", fontsize=8)
        axes[r, 0].axis("off")
        torch.manual_seed(0)
        for c in range(1, 8):
            aug = train_transform(img)
            # Un-normalize for display
            mean, std = _load_norm()
            mean = torch.tensor(mean).view(3, 1, 1)
            std = torch.tensor(std).view(3, 1, 1)
            disp = (aug * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            axes[r, c].imshow(disp)
            axes[r, c].set_title(f"aug #{c}", fontsize=8)
            axes[r, c].axis("off")
    plt.tight_layout()
    out = out_dir / "b7_augmentation_gallery.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("saved:", out)
