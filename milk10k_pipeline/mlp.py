"""
mlp.py
Homework 6: a multi-layer perceptron (MLP) for MILK10k, in PyTorch.

Two kinds of input are supported, with the SAME model and training loop
so that the comparison is fair:

  1. hand-crafted features from Homework 5 (`baseline.load_lesion_table`),
     preprocessed by the same sklearn ColumnTransformer as the baseline
     (median imputation + standardisation + one-hot for sex/site);
  2. raw pixels: each image resized to 32x32x3 and flattened to 3,072
     numbers (`load_pixels`), standardised per channel with train-only
     statistics.

Evaluation protocol (same rules as Homework 5):
  - unit = lesion, Milestone 1 splits;
  - every learned statistic (scaler, class weights) is fitted on train;
  - early stopping uses an INNER 15% split of train (stratified), so the
    official val and test splits are never seen during training and both
    give an unbiased estimate;
  - several seeds per configuration, reported as mean +/- sd, because a
    single run of a small network on 3,668 lesions is noisy.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from . import config
from .baseline import CLASSES

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PIXELS_PATH = config.ROOT / "data" / "processed" / "pixels32.npz"


# ---------------------------------------------------------------------------
# Raw pixels
# ---------------------------------------------------------------------------

def load_pixels(size: int = 32, path=PIXELS_PATH) -> dict[str, np.ndarray]:
    """
    Return {isic_id: uint8 array (size, size, 3)} for every MILK10k image.
    Images are resized with a box/area filter (anti-aliased), which is
    what a 32x32 thumbnail should be. Cached to data/processed/.
    """
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return dict(zip(z["ids"].tolist(), z["pixels"]))
    meta = pd.read_csv(config.METADATA_PATH)
    ids = meta["isic_id"].tolist()
    pix = np.stack([
        np.asarray(Image.open(config.image_path(i)).convert("RGB")
                   .resize((size, size), Image.Resampling.BOX), dtype=np.uint8)
        for i in ids
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids), pixels=pix)
    return dict(zip(ids, pix))


def pixel_matrix(lesions: pd.DataFrame, pixels: dict, views=("derm",)) -> np.ndarray:
    """
    Flatten the 32x32x3 images of each lesion to one row.
    views=("derm",) -> 3,072 numbers; ("derm", "clin") -> 6,144.
    Layout per view is channel-first (C, H, W), like a torch tensor.
    """
    col = {"derm": "derm_id", "clin": "clinical_id"}
    blocks = []
    for v in views:
        arr = np.stack([pixels[i] for i in lesions[col[v]]])           # (n, H, W, 3)
        blocks.append(arr.transpose(0, 3, 1, 2).reshape(len(arr), -1))  # (n, 3*H*W)
    return np.concatenate(blocks, axis=1).astype(np.float32) / 255.0


class ChannelStandardizer:
    """Per-channel mean/std computed on train only (like norm_stats.json)."""

    def __init__(self, n_channels_total: int):
        self.c = n_channels_total

    def fit(self, X):
        Xc = X.reshape(len(X), self.c, -1)
        self.mean_ = Xc.mean(axis=(0, 2))
        self.std_ = Xc.std(axis=(0, 2))
        return self

    def transform(self, X):
        Xc = X.reshape(len(X), self.c, -1)
        Xc = (Xc - self.mean_[None, :, None]) / self.std_[None, :, None]
        return Xc.reshape(len(X), -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Fully-connected network: [Linear -> ReLU -> Dropout] x len(hidden) -> Linear.
    Outputs raw logits for the 3 classes.
    """

    def __init__(self, d_in: int, hidden=(256, 128), p_drop: float = 0.3,
                 n_classes: int = len(CLASSES)):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    hidden: tuple = (256, 128)
    p_drop: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 25
    inner_val_size: float = 0.15
    seed: int = 0


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    inner_val_f1: float
    history: pd.DataFrame = field(repr=False)
    seconds: float = 0.0


def class_weights_from(y: np.ndarray) -> torch.Tensor:
    """weights[i] = n / (n_classes * count_i), same formula as Milestone 1 B8."""
    counts = np.bincount(y, minlength=len(CLASSES))
    return torch.tensor(len(y) / (len(CLASSES) * counts), dtype=torch.float32)


def _macro_f1(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1).numpy()
    return f1_score(y.numpy(), pred, average="macro", labels=range(len(CLASSES)),
                    zero_division=0)


def train_mlp(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> TrainResult:
    """
    Train on X (already standardised) with weighted cross-entropy and AdamW.
    An inner stratified split of the TRAINING data is used for early
    stopping on macro-F1; the best epoch's weights are restored.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    idx_tr, idx_es = train_test_split(np.arange(len(y)), test_size=cfg.inner_val_size,
                                      stratify=y, random_state=cfg.seed)
    Xtr, ytr = torch.tensor(X[idx_tr]), torch.tensor(y[idx_tr])
    Xes, yes = torch.tensor(X[idx_es]), torch.tensor(y[idx_es])

    model = MLP(X.shape[1], cfg.hidden, cfg.p_drop)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_from(y[idx_tr]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(cfg.seed)

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    hist = []
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr), generator=g)
        total = 0.0
        for s in range(0, len(perm), cfg.batch_size):
            b = perm[s:s + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            es_loss = loss_fn(model(Xes), yes).item()
        f1_tr = _macro_f1(model, Xtr, ytr)
        f1_es = _macro_f1(model, Xes, yes)
        hist.append({"epoch": epoch, "train_loss": total / len(ytr), "inner_val_loss": es_loss,
                     "train_macro_f1": f1_tr, "inner_val_macro_f1": f1_es})
        if f1_es > best_f1 + 1e-4:
            best_f1, best_state, best_epoch, bad = f1_es, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, best_epoch, best_f1, pd.DataFrame(hist), time.time() - t0)


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X)), dim=1).numpy()


def encode_labels(labels) -> np.ndarray:
    return np.array([CLASS_TO_IDX[c] for c in labels], dtype=np.int64)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
