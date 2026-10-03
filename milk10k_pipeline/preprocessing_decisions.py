"""
preprocessing_decisions.py
B6: decide the preprocessing strategy and compute train-only stats.

Choices:
  - Input resolution: 224x224.
  - Views: both dermoscopic and clinical per lesion, one tensor each.
  - Normalization: per-channel mean/std computed on the TRAIN split only.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from . import config

OUT = config.CONFIGS_DIR / "norm_stats.json"


def resolution_rationale(out_dir=None):
    """
    Report what fraction of images is larger than a candidate resolution.
    All MILK10k images are 600x450, so this is trivial, but we keep it
    generic so the pipeline works on other datasets.
    """
    sizes_csv = config.FIGURES_DIR / "image_size_summary.csv"
    if not sizes_csv.exists():
        return {"min_width": 600, "median_width": 600, "max_width": 600,
                "min_height": 450, "median_height": 450, "max_height": 450}
    s = pd.read_csv(sizes_csv).set_index("metric")["value"].to_dict()
    return s


def combine_views_table():
    """
    Present the three ways of combining the two views and justify choice (c).
    """
    return pd.DataFrame([
        {
            "option": "a) independent images with the lesion label",
            "pros": "2x effective samples per lesion",
            "cons": "breaks the lesion-level unit; two images of the same lesion can leak across splits",
            "chosen": False,
        },
        {
            "option": "b) one image type only",
            "pros": "simple, no leakage",
            "cons": "discards half the data and loses the complementary information of the clinical view",
            "chosen": False,
        },
        {
            "option": "c) both views per lesion, one tensor each",
            "pros": "uses all the data; the model can learn to combine the two views; matches the lesion-level unit",
            "cons": "needs a model that consumes two inputs (or an aggregation rule)",
            "chosen": True,
        },
    ])


def compute_train_norm_stats(views=("derm", "clinical"), size=(224, 224), max_images=None):
    """
    Compute per-channel mean and std on the TRAIN split only.

    Reads splits/train.csv (lesion-level), loads the corresponding
    images and computes the stats after resizing to `size` and scaling
    to [0, 1].
    """
    train = pd.read_csv(config.SPLITS_DIR / "train.csv")
    if max_images is not None:
        train = train.head(max_images)

    sums = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    n_images = 0

    for _, row in train.iterrows():
        view_col = {"derm": "derm_id", "clinical": "clinical_id"}
        for view in views:
            isic = row[view_col[view]]
            path = config.image_path(isic)
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB").resize(size)
            arr = np.asarray(img, dtype=np.float64) / 255.0
            flat = arr.reshape(-1, 3)
            sums += flat.sum(axis=0)
            sums_sq += (flat ** 2).sum(axis=0)
            n_pixels += flat.shape[0]
            n_images += 1

    mean = sums / n_pixels
    var = sums_sq / n_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0))
    return {
        "views": list(views),
        "size": list(size),
        "n_images": int(n_images),
        "n_pixels": int(n_pixels),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def run(max_images=None):
    print("=" * 72)
    print("B6  Preprocessing decisions")
    print("=" * 72)

    sizes = resolution_rationale()
    print("Image size summary (from B1):")
    for k, v in sizes.items():
        print(f"  {k:15s}: {v}")

    print()
    print("View-combination options:")
    table = combine_views_table()
    print(table.to_string(index=False))

    print()
    print(f"Computing per-channel mean/std on the TRAIN split "
          f"(views={('derm', 'clinical')}, size=224x224)...")
    stats = compute_train_norm_stats(max_images=max_images)
    print(f"  images: {stats['n_images']}")
    print(f"  mean  : {[round(x, 4) for x in stats['mean']]}")
    print(f"  std   : {[round(x, 4) for x in stats['std']]}")

    with open(OUT, "w") as f:
        json.dump(stats, f, indent=2)
    print()
    print("saved:", OUT)
    return stats


if __name__ == "__main__":
    import sys
    max_images = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(max_images=max_images)
