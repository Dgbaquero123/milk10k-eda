"""
splits.py
B5: create and verify the train / val / test splits.

Splits are at the LESION level, grouped by `lesion_id` and stratified
on the 11-class label (`dx`). The three CSV files are written to
`splits/` and committed to the repository.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from . import config

SPLITS_DIR = config.SPLITS_DIR
LABEL_CSV = config.FIGURES_DIR / "lesions_table.csv"
LABEL_MAP = config.CONFIGS_DIR / "label_map.json"


def load_lesions_table():
    """Load the lesion-level table produced in A1.3."""
    path = LABEL_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebook A1.3 first to regenerate it."
        )
    return pd.read_csv(path)


def split_lesions(lesions, val_size=0.15, test_size=0.15, seed=0):
    """
    Deterministic three-way split at the lesion level.

    Groups are lesions (one row each), stratified by `dx`.
    Greedy assignment of each lesion to whichever split is furthest
    below its target size, after a seeded shuffle.
    """
    assert 0 < val_size < 1 and 0 < test_size < 1
    assert val_size + test_size < 1

    n_total = len(lesions)
    target = {
        "train": (1 - val_size - test_size) * n_total,
        "val":   val_size * n_total,
        "test":  test_size * n_total,
    }

    groups = (lesions[["lesion_id", "dx"]]
              .drop_duplicates("lesion_id")
              .sample(frac=1, random_state=seed)
              .reset_index(drop=True))

    tr, va, te = [], [], []
    for _, row in groups.iterrows():
        deficits = {
            "train": target["train"] - len(tr),
            "val":   target["val"]   - len(va),
            "test":  target["test"]  - len(te),
        }
        where = max(deficits, key=deficits.get)
        if where == "train":
            tr.append(row["lesion_id"])
        elif where == "val":
            va.append(row["lesion_id"])
        else:
            te.append(row["lesion_id"])
    return tr, va, te


def write_splits(lesions, train_ids, val_ids, test_ids, out_dir=None):
    """Write three CSV files, one per split, with the lesion-level table."""
    out_dir = Path(out_dir) if out_dir else SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        subset = lesions[lesions["lesion_id"].isin(ids)].reset_index(drop=True)
        out = out_dir / f"{name}.csv"
        subset.to_csv(out, index=False)
        print(f"  {name}: {len(subset)} lesions -> {out}")


def verify_splits(lesions, train_ids, val_ids, test_ids):
    """Print all the B5 required checks and return a summary dict."""
    tr = set(train_ids); va = set(val_ids); te = set(test_ids)

    assert tr.isdisjoint(va), "train and val overlap"
    assert tr.isdisjoint(te), "train and test overlap"
    assert va.isdisjoint(te), "val and test overlap"

    n_total = len(lesions)
    class_of = dict(zip(lesions["lesion_id"], lesions["dx"]))
    diagnosis1_of = dict(zip(lesions["lesion_id"], lesions["diagnosis_1"]))

    def props(ids, class_of):
        s = pd.Series([class_of[i] for i in ids]).value_counts(normalize=True)
        return s

    p_tr = props(train_ids, class_of)
    p_va = props(val_ids,   class_of)
    p_te = props(test_ids,  class_of)
    classes = sorted(set(class_of.values()))
    prop_table = pd.DataFrame({
        "train": p_tr.reindex(classes).fillna(0),
        "val":   p_va.reindex(classes).fillna(0),
        "test":  p_te.reindex(classes).fillna(0),
    }).round(4)
    global_props = pd.Series([class_of[i] for i in lesions["lesion_id"]]).value_counts(normalize=True)
    prop_table["global"] = global_props.reindex(classes).fillna(0).round(4)
    prop_table["max_dev"] = (prop_table[["train", "val", "test"]]
                             .sub(prop_table["global"], axis=0)
                             .abs()
                             .max(axis=1)).round(4)

    p1_tr = pd.Series([diagnosis1_of[i] for i in train_ids]).value_counts(normalize=True).round(4)
    p1_va = pd.Series([diagnosis1_of[i] for i in val_ids]).value_counts(normalize=True).round(4)
    p1_te = pd.Series([diagnosis1_of[i] for i in test_ids]).value_counts(normalize=True).round(4)
    dx1_table = pd.DataFrame({
        "train": p1_tr, "val": p1_va, "test": p1_te,
    }).fillna(0).round(4)

    # Each lesion has exactly 2 images in its split? Check against the
    # metadata file (image-level) by counting per split.
    meta = pd.read_csv(config.METADATA_PATH)
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        sub = meta[meta["lesion_id"].isin(ids)]
        counts = sub.groupby("lesion_id").size()
        assert (counts == 2).all(), f"{name}: some lesions do not have exactly 2 images"
        print(f"  {name}: {counts.sum()} images, all lesions have exactly 2 images")

    # Sizes
    print()
    print("Split sizes:")
    print(f"  train: {len(train_ids):5d} lesions ({len(train_ids)/n_total*100:.2f}%)")
    print(f"  val:   {len(val_ids):5d} lesions ({len(val_ids)/n_total*100:.2f}%)")
    print(f"  test:  {len(test_ids):5d} lesions ({len(test_ids)/n_total*100:.2f}%)")
    print(f"  total: {n_total} lesions")

    print()
    print("Class proportions per split (dx, 11-class scheme):")
    print(prop_table.to_string())
    print(f"  max deviation from global across all splits: "
          f"{prop_table['max_dev'].max():.4f}")

    print()
    print("diagnosis_1 proportions per split:")
    print(dx1_table.to_string())

    return {
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "n_classes": len(classes),
        "max_dev_dx": float(prop_table["max_dev"].max()),
    }


def run():
    print("=" * 72)
    print("B5  Train / val / test splits")
    print("=" * 72)
    lesions = load_lesions_table()
    print(f"lesion table: {lesions.shape}")
    print()

    tr, va, te = split_lesions(lesions, val_size=0.15, test_size=0.15, seed=config.SEED)
    print("Splits created. Writing CSVs:")
    write_splits(lesions, tr, va, te)
    print()
    print("Verification:")
    verify_splits(lesions, tr, va, te)

    # Persist the split metadata in the label_map.json
    label_map_path = LABEL_MAP
    if label_map_path.exists():
        with open(label_map_path) as f:
            lm = json.load(f)
        lm["split_seed"] = config.SEED
        lm["split_date"] = config.SPLIT_DATE
        lm["split_sizes"] = {
            "train": len(tr), "val": len(va), "test": len(te),
        }
        with open(label_map_path, "w") as f:
            json.dump(lm, f, indent=2)
        print()
        print(f"label_map.json updated with split info: {label_map_path}")


if __name__ == "__main__":
    run()
