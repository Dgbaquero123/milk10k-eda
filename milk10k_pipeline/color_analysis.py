"""
color_analysis.py
Part 2: dataset level color and histogram analysis.

Functions
---------
load_image_rgb(path)
load_image_gray(path)
sample_paths_per_class(df, images_dir, n_per_class, seed)
average_gray_histogram_per_class(paths_by_class, bins)
average_rgb_histogram_per_class(paths_by_class, bins)
per_image_color_stats(paths_by_class)
plot_average_gray_histogram(hists, out_path)
plot_average_rgb_histogram(hists, out_path)
plot_color_stats_boxplots(stats_df, out_path)
ks_overlap_matrix(stats_df, channels)
relative_mean_gap(stats_df, channels)
run_part2(df, images_dir, figures_dir, n_per_class, seed)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2

TARGET = "diagnosis_1"


def load_image_rgb(path):
    """Load an image as an RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_image_gray(path):
    """Load an image as a grayscale numpy array."""
    arr = load_image_rgb(path)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def sample_paths_per_class(df, images_dir, n_per_class=25, seed=0):
    """
    Return a dict {class_label: [paths]} with at most n_per_class paths
    per class, drawn from images that exist locally.
    """
    images_dir = Path(images_dir)
    rng = np.random.default_rng(seed)
    out = {}
    for cls, g in df.groupby(TARGET):
        paths = []
        for isic in g["isic_id"]:
            p = images_dir / f"{isic}.jpg"
            if p.exists():
                paths.append(p)
        if len(paths) > n_per_class:
            idx = rng.choice(len(paths), size=n_per_class, replace=False)
            paths = [paths[i] for i in idx]
        out[cls] = paths
    return out


def average_gray_histogram_per_class(paths_by_class, bins=64):
    """Normalized average grayscale histogram per class."""
    hists = {}
    for cls, paths in paths_by_class.items():
        acc = np.zeros(bins)
        n_ok = 0
        for p in paths:
            try:
                g = load_image_gray(p)
                h, _ = np.histogram(g.ravel(), bins=bins, range=(0, 255))
                acc += h / h.sum()
                n_ok += 1
            except Exception:
                continue
        if n_ok > 0:
            acc /= n_ok
        hists[cls] = acc
    return hists


def average_rgb_histogram_per_class(paths_by_class, bins=64):
    """Normalized average per channel RGB histogram per class."""
    hists = {}
    for cls, paths in paths_by_class.items():
        acc = np.zeros((3, bins))
        n_ok = 0
        for p in paths:
            try:
                arr = load_image_rgb(p)
            except Exception:
                continue
            n_ok += 1
            for c in range(3):
                h, _ = np.histogram(arr[:, :, c].ravel(), bins=bins, range=(0, 255))
                acc[c] += h / h.sum()
        if n_ok > 0:
            acc /= n_ok
        hists[cls] = acc
    return hists


def per_image_color_stats(paths_by_class):
    """Per image mean and std of each RGB channel, plus the class label."""
    rows = []
    for cls, paths in paths_by_class.items():
        for p in paths:
            try:
                arr = load_image_rgb(p).astype(np.float32)
            except Exception:
                continue
            row = {"class": cls, "isic_id": p.stem}
            for i, ch in enumerate(["R", "G", "B"]):
                row[f"{ch}_mean"] = arr[:, :, i].mean()
                row[f"{ch}_std"] = arr[:, :, i].std()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_average_gray_histogram(hists, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, h in hists.items():
        ax.plot(np.linspace(0, 255, len(h)), h, label=cls)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Normalized frequency")
    ax.set_title("Average grayscale histogram per class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_average_rgb_histogram(hists, out_path):
    fig, axes = plt.subplots(1, len(hists), figsize=(5 * len(hists), 4), sharey=True)
    if len(hists) == 1:
        axes = [axes]
    colors = ["red", "green", "blue"]
    for ax, (cls, h) in zip(axes, hists.items()):
        for c in range(3):
            ax.plot(np.linspace(0, 255, h.shape[1]), h[c],
                    color=colors[c], label=["R", "G", "B"][c])
        ax.set_title(cls)
        ax.set_xlabel("Intensity")
        ax.legend()
    axes[0].set_ylabel("Normalized frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_color_stats_boxplots(df_stats, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, ch in enumerate(["R", "G", "B"]):
        df_stats.boxplot(column=f"{ch}_mean", by="class", ax=axes[0, i])
        axes[0, i].set_title(f"{ch} mean per class")
        axes[0, i].set_xlabel("")
        df_stats.boxplot(column=f"{ch}_std", by="class", ax=axes[1, i])
        axes[1, i].set_title(f"{ch} std per class")
        axes[1, i].set_xlabel("")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def ks_overlap_matrix(stats_df, channels=("R_mean", "G_mean", "B_mean")):
    """
    Pairwise Kolmogorov-Smirnov comparison of per channel means between
    classes. Returns a DataFrame with one row per channel and class pair.
    """
    from scipy.stats import ks_2samp
    import itertools

    classes = sorted(stats_df["class"].unique())
    pairs = list(itertools.combinations(classes, 2))
    rows = []
    for ch in channels:
        for a, b in pairs:
            xa = stats_df.loc[stats_df["class"] == a, ch].values
            xb = stats_df.loc[stats_df["class"] == b, ch].values
            if len(xa) < 2 or len(xb) < 2:
                continue
            stat, p = ks_2samp(xa, xb)
            rows.append({"channel": ch, "class_a": a, "class_b": b,
                         "ks_stat": round(stat, 4), "p_value": p})
    return pd.DataFrame(rows)


def relative_mean_gap(stats_df, channels=("R_mean", "G_mean", "B_mean")):
    """
    Relative difference between the maximum and minimum class mean for
    each channel, expressed as a percentage of the global mean.
    """
    rows = []
    for ch in channels:
        by_class = stats_df.groupby("class")[ch].mean()
        global_mean = by_class.mean()
        gap_pct = 100.0 * (by_class.max() - by_class.min()) / global_mean
        rows.append({
            "channel": ch,
            "min_class_mean": round(by_class.min(), 2),
            "max_class_mean": round(by_class.max(), 2),
            "global_mean": round(global_mean, 2),
            "gap_pct": round(gap_pct, 2),
        })
    return pd.DataFrame(rows)


def run_part2(df, images_dir, figures_dir, n_per_class=25, seed=0):
    """
    Run the full Part 2 pipeline: sample images, compute histograms and
    color statistics, and save the figures and the CSV summary.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths_by_class = sample_paths_per_class(df, images_dir,
                                            n_per_class=n_per_class, seed=seed)
    gray_h = average_gray_histogram_per_class(paths_by_class)
    rgb_h = average_rgb_histogram_per_class(paths_by_class)
    stats_df = per_image_color_stats(paths_by_class)

    plot_average_gray_histogram(gray_h, figures_dir / "part2_gray_hist_per_class.png")
    plot_average_rgb_histogram(rgb_h, figures_dir / "part2_rgb_hist_per_class.png")
    plot_color_stats_boxplots(stats_df, figures_dir / "part2_color_stats_boxplots.png")
    stats_df.to_csv(figures_dir / "part2_color_stats.csv", index=False)

    return {
        "paths_by_class": paths_by_class,
        "gray_hists": gray_h,
        "rgb_hists": rgb_h,
        "stats_df": stats_df,
    }
