"""
dataset.py
B8: lesion-level Dataset and DataLoaders for MILK10k.

Design choices:
  - Each item is ONE lesion. It returns a dict with one tensor per
    requested view, the integer label, and the lesion_id.
  - train uses `train_transform` (augmented); val/test use
    `eval_transform` (deterministic).
  - Class weights are computed from the TRAIN split only and saved to
    `configs/class_weights.json`.
  - A `WeightedRandomSampler` is provided for train as an alternative
    to class-weighted loss.
  - Missing files raise FileNotFoundError (fail loudly).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from . import config
from .transforms import train_transform, eval_transform

VIEW_COL = {"derm": "derm_id", "clinical": "clinical_id"}


class LesionDataset(Dataset):
    """
    One item = one lesion.

    Parameters
    ----------
    split_df : pd.DataFrame with columns
        lesion_id, derm_id, clinical_id, diagnosis_1, dx
    label_col : str, which column to use as the class label
    class_to_idx : dict mapping class name -> int
    transform : torchvision transform
    views : tuple, e.g. ("derm", "clinical")
    allow_missing : bool, if True skip missing images, else raise
    """

    def __init__(self, split_df, label_col="diagnosis_1",
                 class_to_idx=None, transform=None,
                 views=("derm", "clinical"), allow_missing=False):
        self.df = split_df.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform if transform is not None else eval_transform
        self.views = tuple(views)
        self.allow_missing = allow_missing

        if class_to_idx is None:
            self.classes = sorted(self.df[label_col].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = [None] * len(class_to_idx)
            for c, i in class_to_idx.items():
                self.classes[i] = c

    def __len__(self):
        return len(self.df)

    def _load_one(self, isic_id):
        path = config.image_path(isic_id)
        if not path.exists():
            if self.allow_missing:
                return None
            raise FileNotFoundError(f"Missing image: {path}")
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        for v in self.views:
            t = self._load_one(row[VIEW_COL[v]])
            if t is None:
                return None  # loader will skip via collate_fn
            item[v] = t
        item["label"] = int(self.class_to_idx[row[self.label_col]])
        item["lesion_id"] = str(row["lesion_id"])
        return item


def _collate_skip_none(batch):
    """Drop items that came back as None (only used when allow_missing=True)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


def load_splits(label_col="diagnosis_1"):
    """Load the 3 split CSVs and build a single class_to_idx mapping."""
    train = pd.read_csv(config.SPLITS_DIR / "train.csv")
    val = pd.read_csv(config.SPLITS_DIR / "val.csv")
    test = pd.read_csv(config.SPLITS_DIR / "test.csv")
    classes = sorted(set(train[label_col]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return train, val, test, classes, class_to_idx


def compute_class_weights(train_df, label_col="diagnosis_1",
                          class_to_idx=None):
    """
    Compute per-class weights inversely proportional to frequency.
    weights[i] = n_total / (n_classes * count_i)
    """
    if class_to_idx is None:
        classes = sorted(train_df[label_col].unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(class_to_idx)
    n_total = len(train_df)
    counts = train_df[label_col].value_counts().to_dict()
    weights = [0.0] * n_classes
    for c, i in class_to_idx.items():
        weights[i] = n_total / (n_classes * counts[c])
    return weights


def build_loaders(batch_size=16, num_workers=0, use_sampler=False,
                  views=("derm", "clinical"), label_col="diagnosis_1"):
    """
    Build the three DataLoaders and save class weights to disk.
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    train_df, val_df, test_df, classes, class_to_idx = load_splits(label_col=label_col)
    weights = compute_class_weights(train_df, label_col=label_col,
                                    class_to_idx=class_to_idx)

    # Save class weights
    out = config.CONFIGS_DIR / "class_weights.json"
    with open(out, "w") as f:
        json.dump({
            "classes": classes,
            "class_to_idx": class_to_idx,
            "weights": weights,
            "label_col": label_col,
        }, f, indent=2)

    train_ds = LesionDataset(train_df, label_col=label_col,
                             class_to_idx=class_to_idx,
                             transform=train_transform, views=views)
    val_ds = LesionDataset(val_df, label_col=label_col,
                           class_to_idx=class_to_idx,
                           transform=eval_transform, views=views)
    test_ds = LesionDataset(test_df, label_col=label_col,
                            class_to_idx=class_to_idx,
                            transform=eval_transform, views=views)

    sampler = None
    if use_sampler:
        labels = train_df[label_col].map(class_to_idx).values
        sample_weights = np.array([weights[i] for i in labels])
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=num_workers,
                              collate_fn=_collate_skip_none)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            collate_fn=_collate_skip_none)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers,
                             collate_fn=_collate_skip_none)
    return train_loader, val_loader, test_loader, (classes, class_to_idx, weights)


def sanity_check(loader, n_batches=20, label="train"):
    """Print batch shape, dtype, min/max, label histogram over n_batches."""
    print(f"[{label}] iterating {n_batches} batches...")
    label_counts = {}
    first_batch_shape = None
    n_seen = 0
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        if first_batch_shape is None:
            first_batch_shape = {k: tuple(v.shape) for k, v in batch.items()
                                 if torch.is_tensor(v)}
            for k, v in batch.items():
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    print(f"  batch['{k}']: shape={tuple(v.shape)}, "
                          f"dtype={v.dtype}, min={v.min():.3f}, max={v.max():.3f}")
        for lbl in batch["label"].tolist():
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        n_seen += 1
        if n_seen >= n_batches:
            break
    print(f"  label histogram over {n_batches} batches: {label_counts}")


def run():
    print("=" * 72)
    print("B8  Dataset and DataLoaders")
    print("=" * 72)
    train_loader, val_loader, test_loader, (classes, class_to_idx, weights) = \
        build_loaders(batch_size=16, num_workers=0)

    print(f"classes       : {classes}")
    print(f"class_to_idx  : {class_to_idx}")
    print(f"class_weights : {[round(w, 3) for w in weights]}")
    print()

    import time
    t0 = time.time()
    sanity_check(train_loader, n_batches=20, label="train")
    t1 = time.time()
    print(f"  time for 20 train batches: {t1 - t0:.2f}s")
    print()
    sanity_check(val_loader, n_batches=5, label="val")


if __name__ == "__main__":
    run()
