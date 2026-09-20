"""
visualizer.py
Parte 5: utilidades de visualizacion.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = "diagnosis_1"


def show_grid(images, labels=None, ncols=5, title=None, out_path=None):
    """Muestra un grid de imagenes con sus etiquetas como titulos."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]
            if img.dtype != np.uint8 and img.max() <= 1.5:
                img_show = np.clip(img, 0, 1)
            else:
                img_show = img
            ax.imshow(img_show, cmap="gray" if img.ndim == 2 else None)
            if labels is not None and i < len(labels):
                ax.set_title(str(labels[i]), fontsize=9)
            ax.axis("off")
        else:
            ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        plt.close()
    return fig


def plot_class_balance(df, out_path=None):
    """Bar chart de balance de clases."""
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Balance de clases (diagnosis_1)")
    ax.set_ylabel("Numero de imagenes")
    ax.set_xlabel("")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        plt.close()
    return fig, counts


def plot_batch_summary(X, out_path=None):
    """
    Dado un batch (B, H, W, C) o (B, H, W), muestra:
      - distribucion de pixeles (histograma)
      - min/max/mean del batch
    """
    flat = X.reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat, bins=60, color="purple", alpha=0.8)
    ax.set_title(f"Distribucion de pixeles del batch  |  min={flat.min():.3f}  max={flat.max():.3f}  mean={flat.mean():.3f}")
    ax.set_xlabel("Valor de pixel")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        plt.close()
    return fig


def plot_raw_vs_processed(raw, processed, out_path=None):
    """Compara una imagen cruda (uint8) vs procesada (float) en 2 subplots."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw)
    axes[0].set_title("Raw")
    axes[0].axis("off")
    img = processed
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    axes[1].imshow(img, cmap="gray" if img.ndim == 2 else None)
    axes[1].set_title("Procesada")
    axes[1].axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        plt.close()
    return fig
