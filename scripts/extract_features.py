"""
extract_features.py
Homework 5: compute hand-crafted features for every MILK10k image once
and cache them, so model comparison does not re-read 10,480 JPEGs.

Usage (from the repository root):
    python scripts/extract_features.py            # all images
    python scripts/extract_features.py --limit 50 # quick smoke test

Output: data/processed/image_features.csv.gz (one row per isic_id).
The file is gitignored; re-run this script to regenerate it.
"""
import argparse
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from milk10k_pipeline import config               # noqa: E402
from milk10k_pipeline.features import image_features  # noqa: E402

OUT = ROOT / "data" / "processed" / "image_features.csv.gz"


def _one(isic_id):
    feats = image_features(config.image_path(isic_id))
    feats["isic_id"] = isic_id
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=cpu_count())
    args = ap.parse_args()

    meta = pd.read_csv(config.METADATA_PATH)
    ids = meta["isic_id"].tolist()[: args.limit]
    print(f"Extracting features for {len(ids)} images with {args.workers} workers")

    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, ids, chunksize=16), 1):
            rows.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows).set_index("isic_id").loc[ids].reset_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"Saved {df.shape} to {OUT.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
