"""
config.py
Single source of configuration for the project.

All paths are relative to the repository root, except the images directory
which can be overridden with the MILK10K_IMAGES_DIR environment variable.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
ROOT = next(p for p in [_HERE.parent.parent] + list(_HERE.parents) if (p / "data").is_dir())

DATA_DIR = ROOT / "data" / "raw" / "milk10k"
METADATA_PATH = DATA_DIR / "metadata.csv"
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"
TRAIN_INPUT_PATH = DATA_DIR / "supplements" / "training_input.csv"
TRAIN_SUPP_PATH = DATA_DIR / "supplements" / "training_supp.csv"

IMG_DIR = Path(os.environ.get("MILK10K_IMAGES_DIR", DATA_DIR / "images"))

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures_task3"
SPLITS_DIR = ROOT / "splits"
CONFIGS_DIR = ROOT / "configs"

for _d in (REPORTS_DIR, FIGURES_DIR, SPLITS_DIR, CONFIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 0
IMAGE_SIZE = (224, 224)
SPLIT_DATE = "2026-10-03"
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_path(isic_id: str) -> Path:
    """Return the full path of the .jpg image for the given isic_id."""
    return IMG_DIR / f"{isic_id}.jpg"


def sanity_check() -> None:
    """Print the resolved configuration. Use to verify paths at runtime."""
    print("ROOT             :", ROOT)
    print("METADATA_PATH    :", METADATA_PATH, "exists:", METADATA_PATH.exists())
    print("GT_PATH          :", GT_PATH, "exists:", GT_PATH.exists())
    print("IMG_DIR          :", IMG_DIR, "exists:", IMG_DIR.exists())
    print("FIGURES_DIR      :", FIGURES_DIR)
    print("SPLITS_DIR       :", SPLITS_DIR)
    print("SEED             :", SEED)
    print("IMAGE_SIZE       :", IMAGE_SIZE)


if __name__ == "__main__":
    sanity_check()
