# MILK10k — Skin Lesion Classification

## 1. What this project is

**Task.** Classify skin lesions into three diagnosis groups
(`Benign` / `Malignant` / `Indeterminate`) from pairs of dermatological
images: one dermoscopic and one clinical close-up per lesion, optionally
combined with demographic metadata. A **stretch goal** is the 11-class
scheme used in `training_gt.csv` (BCC, NV, BKL, SCCKA, MEL, AKIEC, DF,
INF, VASC, BEN_OTH, MAL_OTH).

**Dataset.** [MILK10k](https://github.com/DIAGNijmegen/MILK10k) from the
DIAG Nijmegen group. It contains 10,480 images covering 5,240 lesions
(exactly two views per lesion), 17 metadata columns, and a per-lesion
one-hot label file `training_gt.csv`. The images and metadata are
distributed under CC-BY-NC; the raw images are not committed to this
repo, only the metadata CSVs and the code.

**Milestone 1 goal.** Turn the Session 2 homework code into a
**leak-free, reproducible** project pipeline: integrity-checked images,
lesion-level splits grouped by `lesion_id`, a label mapping saved to
disk, a reusable augmentation module, and a lesion-level
`Dataset`/`DataLoader` ready for training in Milestone 2.

## 2. How to set up and run

**Python version.** 3.13.

**Install dependencies.**


**Where to put the data.** The raw images and metadata should live under
`data/raw/milk10k/`. The code reads the folder from a single variable in
`milk10k_pipeline/config.py` and from the `MILK10K_IMAGES_DIR`
environment variable (override it if your images live elsewhere):


**Reproduce the splits, figures and tests.**


## 3. Repository structure


**Why this structure.** Reusable logic lives inside the importable
`milk10k_pipeline/` package and is never copied between notebooks.
Exploration and homework answers live in `notebooks/` so they can be
exported to PDF without dragging package code with them. Generated
outputs (figures, splits, JSON configs, reports) live outside the
package in `reports/`, `splits/` and `configs/`, so the source tree
stays clean and `git clean` does not delete them accidentally. The
raw dataset is the only thing not committed.

## 4. Data handling rules

- **Committed.** Metadata CSVs (`data/raw/milk10k/metadata.csv`,
  `supplements/*.csv`), split CSVs (`splits/*.csv`), label map,
  class weights, normalization stats, all code, all figures under
  `reports/` that are small enough.
- **Not committed.** The raw images (`data/raw/milk10k/images/`) are
  gitignored; the report PDF (`reports/milestone1_report.pdf`) is
  gitignored as well because it is generated.
- **Where generated files go.** `reports/figures_milestone1/` for
  figures, `splits/` for the split CSVs, `configs/` for JSON configs.
- **Seed and split date.** Splits were created with `seed=0` on
  `2026-10-03`. Both are recorded in `milk10k_pipeline/config.py`
  (`SEED`, `SPLIT_DATE`) and in `configs/label_map.json`.

## 5. Key decisions and results so far

- **Label strategy.** The primary target is `diagnosis_1` with three
  classes (Benign / Malignant / Indeterminate). `AKIEC` is the only
  11-class label that does not map cleanly to a single `diagnosis_1`
  (it splits into 360 Malignant and 246 Indeterminate lesions); for the
  stretch goal, this is handled by treating the 3-class task as primary
  and the 11-class scheme as a refinement of it. See A1.1 in the Part A
  notebook for the full cross-check.
- **Split design.** Lesion-level, grouped by `lesion_id` and stratified
  on the 11-class label. `StratifiedGroupKFold` was chosen after
  comparing it with `GroupKFold` and image-level `StratifiedKFold`:
  the latter leaked 4,235 lesions across folds, while
  `StratifiedGroupKFold` had zero leak and a maximum class-proportion
  spread of 0.10 pp across folds.
- **Preprocessing.** Min-max normalization to `[0, 1]`, image size
  224×224, RGB. Two views per lesion. Per-channel mean/std are computed
  on the train split only and saved to `configs/norm_stats.json`.
- **Imbalance handling.** Class weights computed from the train split
  and saved to `configs/class_weights.json`; a `WeightedRandomSampler`
  is provided as an alternative. Both options are implemented in B8.

See `reports/milestone1_report.pdf` for the full write-up.
