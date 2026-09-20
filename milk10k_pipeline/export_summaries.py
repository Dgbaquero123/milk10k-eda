"""
export_summaries.py
Generate summary tables and documentation for the Task 2 report:

  - field usage recommendation table
  - Milk10kLoader interface documentation
  - color overlap between classes (KS test and relative mean gap)

Outputs are written to reports/figures_task2/.
"""
from pathlib import Path
import pandas as pd

from .metadata_analysis import load_metadata
from .color_analysis import (sample_paths_per_class,
                             per_image_color_stats,
                             ks_overlap_matrix,
                             relative_mean_gap)


# ---------------------------------------------------------------------------
# Field usage recommendation
# ---------------------------------------------------------------------------

FIELD_USAGE = [
    ("isic_id",                "identifier",  "do not use as a feature"),
    ("lesion_id",              "identifier",  "do not use as a feature; group splits by it"),
    ("attribution",            "constant",    "uninformative"),
    ("copyright_license",      "constant",    "uninformative"),
    ("age_approx",             "numeric",     "use, with imputation"),
    ("anatom_site_general",    "categorical", "use with caution; proxy for clinical site"),
    ("anatom_site_special",    "categorical", "avoid; 98 percent missing"),
    ("concomitant_biopsy",     "boolean",     "use; informative"),
    ("diagnosis_1",            "target",      "target"),
    ("diagnosis_2",            "categorical", "DO NOT USE; hierarchical leakage"),
    ("diagnosis_3",            "categorical", "DO NOT USE; hierarchical leakage"),
    ("diagnosis_4",            "categorical", "DO NOT USE; hierarchical leakage"),
    ("diagnosis_confirm_type", "categorical", "use with caution; protocol bias"),
    ("image_manipulation",     "categorical", "use with caution; device proxy"),
    ("image_type",             "categorical", "uninformative; no variation"),
    ("melanocytic",            "categorical", "uninformative; 77 percent missing, one value"),
    ("sex",                    "categorical", "use; weak effect, Cramer's V about 0.09"),
]


def build_field_table():
    return pd.DataFrame(FIELD_USAGE,
                        columns=["column", "type", "recommendation"])


# ---------------------------------------------------------------------------
# Loader interface documentation
# ---------------------------------------------------------------------------

LOADER_DOC = """\
Milk10kLoader
=============

Constructor
-----------
Milk10kLoader(
    metadata,               # pd.DataFrame read from metadata.csv
    images_dir,             # Path to the folder with .jpg images
    batch_size=16,          # number of images per batch
    size=(224, 224),        # output size (height, width)
    color_space="rgb",      # 'rgb' or 'gray'
    normalization="minmax", # 'minmax', 'zscore' or None
    shuffle=True,           # shuffle row order between epochs
    seed=0,                 # seed used for reproducible shuffle
)

Public attributes
-----------------
loader.classes   -> sorted list of unique classes in the available subset
loader.df        -> DataFrame filtered to rows with an existing image on disk
len(loader)      -> number of batches per epoch

Iteration
---------
for X, y in loader:
    # X: np.ndarray of shape (B, H, W, C), float32, normalized
    # y: np.ndarray of strings with the diagnosis_1 labels

Behavior
--------
- Automatically filters out rows whose file is missing from images_dir.
- Applies the Part 3 preprocess_image function per image when the batch is
  produced, so the full dataset is never loaded into memory at once.
- If a single image fails to load, it is skipped and the batch continues.
"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(metadata_path, figures_dir, images_dir):
    """Write all summary tables and documents into figures_dir."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fields = build_field_table()
    fields.to_csv(figures_dir / "summary_field_recommendations.csv", index=False)

    (figures_dir / "summary_loader_interface.txt").write_text(LOADER_DOC)

    df = load_metadata(metadata_path)
    paths = sample_paths_per_class(df, images_dir, n_per_class=25, seed=0)
    stats_df = per_image_color_stats(paths)

    ks = ks_overlap_matrix(stats_df)
    gaps = relative_mean_gap(stats_df)
    ks.to_csv(figures_dir / "summary_color_ks.csv", index=False)
    gaps.to_csv(figures_dir / "summary_color_gaps.csv", index=False)

    return fields, ks, gaps
