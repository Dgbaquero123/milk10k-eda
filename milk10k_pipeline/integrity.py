"""
integrity.py
B1: full-dataset integrity check.

Walks every row of metadata.csv, verifies that the corresponding image
exists and can be decoded (PIL.Image.verify()), and writes a summary
table with the width and height distribution.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from . import config


def check_all_images(df, img_dir, verbose_every=500):
    """
    Verify every image referenced by df['isic_id'] under img_dir.

    Returns
    -------
    missing : list of isic_id with no file on disk
    unreadable : list of (isic_id, error_message) for files that fail verify()
    sizes : list of (isic_id, width, height)
    """
    missing = []
    unreadable = []
    sizes = []
    n = len(df)
    for k, isic_id in enumerate(df["isic_id"]):
        path = img_dir / f"{isic_id}.jpg"
        if not path.exists():
            missing.append(isic_id)
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            # verify() leaves the file unusable; reopen to read the size
            with Image.open(path) as im:
                w, h = im.size
            sizes.append((isic_id, w, h))
        except (UnidentifiedImageError, OSError) as e:
            unreadable.append((isic_id, str(e)))
        if verbose_every and (k + 1) % verbose_every == 0:
            print(f"  checked {k + 1}/{n}")
    return missing, unreadable, sizes


def run(metadata_path=None, img_dir=None, out_dir=None, fail_loud=True):
    """Run the full integrity check and write the summary table."""
    metadata_path = Path(metadata_path) if metadata_path else config.METADATA_PATH
    img_dir = Path(img_dir) if img_dir else config.IMG_DIR
    out_dir = Path(out_dir) if out_dir else config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("B1  Full-dataset integrity check")
    print("=" * 72)
    print(f"metadata : {metadata_path}")
    print(f"images   : {img_dir}")
    print(f"output   : {out_dir}")
    print()

    df = pd.read_csv(metadata_path)
    print(f"rows in metadata: {len(df)}")
    print(f"unique isic_id  : {df['isic_id'].nunique()}")
    print()

    missing, unreadable, sizes = check_all_images(df, img_dir)

    print()
    print(f"missing files      : {len(missing)}")
    if missing:
        for isic in missing[:20]:
            print(f"  - {isic}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    print(f"unreadable files   : {len(unreadable)}")
    for isic, msg in unreadable[:20]:
        print(f"  - {isic}: {msg}")

    if sizes:
        sizes_df = pd.DataFrame(sizes, columns=["isic_id", "width", "height"])
        summary = pd.DataFrame({
            "metric": ["count", "min_width", "median_width", "max_width",
                       "min_height", "median_height", "max_height",
                       "unique_sizes"],
            "value": [
                len(sizes_df),
                int(sizes_df["width"].min()),
                int(sizes_df["width"].median()),
                int(sizes_df["width"].max()),
                int(sizes_df["height"].min()),
                int(sizes_df["height"].median()),
                int(sizes_df["height"].max()),
                sizes_df.groupby(["width", "height"]).ngroups,
            ],
        })
        print()
        print("size distribution:")
        print(summary.to_string(index=False))

        # Save the summary and a per-size count table
        out_csv = out_dir / "image_size_summary.csv"
        summary.to_csv(out_csv, index=False)
        per_size = (sizes_df.groupby(["width", "height"]).size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False))
        per_size.to_csv(out_dir / "image_size_counts.csv", index=False)
        print()
        print(f"saved: {out_csv}")
        print(f"saved: {out_dir / 'image_size_counts.csv'}")

    # Fail loudly if requested
    if fail_loud:
        if missing:
            raise SystemExit(f"FAIL: {len(missing)} images are missing")
        if unreadable:
            raise SystemExit(f"FAIL: {len(unreadable)} images are unreadable")

    print()
    print("integrity check passed.")
    return summary if sizes else None


if __name__ == "__main__":
    run()
