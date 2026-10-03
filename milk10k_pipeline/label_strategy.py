"""
label_strategy.py
B3: decide the label strategy for Milestone 1 and write configs/label_map.json.

Primary target
--------------
diagnosis_1 with 3 classes: Benign / Indeterminate / Malignant.
Indeterminate is kept as its own class, not merged or dropped, because
(i) it has 123 lesions, which is enough to be learnable, and (ii) in a
clinical workflow it maps to a distinct action (refer for biopsy), so
losing it would collapse a real decision boundary into a wrong one.

Stretch target
--------------
The 11-class scheme from training_gt.csv. Classes with fewer than ~50
lesions are merged into a single 'other' bucket, because a class with
9 lesions cannot be reliably learned or evaluated on a single split.
Classes above the threshold are kept as their own class.
"""
import json
from pathlib import Path
import pandas as pd

from . import config


RARE_THRESHOLD = 50  # lesions; below this, merge into "other"


PRIMARY_CLASSES = ["Benign", "Indeterminate", "Malignant"]
PRIMARY_TO_IDX = {c: i for i, c in enumerate(PRIMARY_CLASSES)}


def build_label_map(out_path=None):
    """
    Build and save the label mapping for the project.

    The JSON has three sections:
      primary      : diagnosis_1 -> integer index (3 classes)
      stretch_11   : the raw 11-class labels, with counts, and their
                     "kept" / "merged" fate
      stretch_final: the final integer index for the stretch task after
                     merging rare classes into "other"
    """
    out_path = Path(out_path) if out_path else config.CONFIGS_DIR / "label_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the per-lesion 11-class label
    gt = pd.read_csv(config.GT_PATH)
    class_cols = [c for c in gt.columns if c != "lesion_id"]
    gt_long = gt.melt(id_vars="lesion_id", value_vars=class_cols,
                      var_name="dx", value_name="is_dx")
    gt_long = gt_long[gt_long["is_dx"] == 1][["lesion_id", "dx"]]
    counts = gt_long["dx"].value_counts().sort_values(ascending=False)

    # Decide fate of each class
    kept = [c for c in counts.index if counts[c] >= RARE_THRESHOLD]
    merged = [c for c in counts.index if counts[c] < RARE_THRESHOLD]

    # Final stretch classes: kept classes + one "other" bucket
    stretch_final = kept + ["other"]
    stretch_to_idx = {c: i for i, c in enumerate(stretch_final)}

    label_map = {
        "primary_classes": PRIMARY_CLASSES,
        "primary_to_idx": PRIMARY_TO_IDX,
        "stretch_11_classes": list(counts.index),
        "stretch_11_counts": {c: int(counts[c]) for c in counts.index},
        "stretch_threshold": RARE_THRESHOLD,
        "stretch_kept": kept,
        "stretch_merged": merged,
        "stretch_final_classes": stretch_final,
        "stretch_final_to_idx": stretch_to_idx,
        "seed": config.SEED,
        "split_date": config.SPLIT_DATE,
    }

    with open(out_path, "w") as f:
        json.dump(label_map, f, indent=2)
    return label_map, out_path


def run():
    print("=" * 72)
    print("B3  Label strategy")
    print("=" * 72)
    label_map, out_path = build_label_map()

    print("primary classes     :", label_map["primary_classes"])
    print("primary_to_idx      :", label_map["primary_to_idx"])
    print()
    print(f"stretch threshold   : {label_map['stretch_threshold']} lesions")
    print("stretch kept        :", label_map["stretch_kept"])
    print("stretch merged      :", label_map["stretch_merged"])
    print("stretch final classes:", label_map["stretch_final_classes"])
    print()
    print("11-class counts:")
    for c, n in label_map["stretch_11_counts"].items():
        tag = "kept" if c in label_map["stretch_kept"] else "merged into 'other'"
        print(f"  {c:8s} {n:5d}   ({tag})")
    print()
    print("saved:", out_path)


if __name__ == "__main__":
    run()
