# Submission — Homework Sessions 1-3 + Course Project Milestone 1

**Student:** Daniela Gonzalez
**Repository:** https://github.com/Dgbaquero123/milk10k-eda

## Part A — Sessions 1, 2, 3 homework

- Notebook (source): [notebooks/homework_part_a.ipynb](notebooks/homework_part_a.ipynb)
- Notebook (PDF): [notebooks/homework_part_a.pdf](notebooks/homework_part_a.pdf) — to be exported before submission

## Part B — Course Project Milestone 1

- README: [README.md](README.md)
- Requirements: [requirements.txt](requirements.txt)
- Package: [milk10k_pipeline/](milk10k_pipeline/)

### Source code

- Configuration: [milk10k_pipeline/config.py](milk10k_pipeline/config.py)
- Integrity check: [milk10k_pipeline/integrity.py](milk10k_pipeline/integrity.py)
- Label strategy: [milk10k_pipeline/label_strategy.py](milk10k_pipeline/label_strategy.py)
- Quality report: [milk10k_pipeline/quality_report.py](milk10k_pipeline/quality_report.py)
- Splits: [milk10k_pipeline/splits.py](milk10k_pipeline/splits.py)
- Transforms: [milk10k_pipeline/transforms.py](milk10k_pipeline/transforms.py)
- Dataset / DataLoader: [milk10k_pipeline/dataset.py](milk10k_pipeline/dataset.py)

### Configs and splits

- Label map: [configs/label_map.json](configs/label_map.json)
- Class weights: [configs/class_weights.json](configs/class_weights.json)
- Normalization stats: [configs/norm_stats.json](configs/norm_stats.json)
- Train split: [splits/train.csv](splits/train.csv)
- Val split: [splits/val.csv](splits/val.csv)
- Test split: [splits/test.csv](splits/test.csv)

### Figures and reports

- Milestone 1 figures: [reports/figures_milestone1/](reports/figures_milestone1/)
- Data-quality report: [reports/quality_report.md](reports/quality_report.md)
- Milestone 1 report: [reports/milestone1_report.pdf](reports/milestone1_report.pdf) — to be generated before submission

### Tests

- Split property tests: [tests/test_splits.py](tests/test_splits.py)
- Dataset smoke tests: [tests/test_dataset.py](tests/test_dataset.py)

## Homework 4 — Filters as Features

- Notebook: [notebooks/homework_part_4.ipynb](notebooks/homework_part_4.ipynb)
- Figures: [reports/figures_hw4/](reports/figures_hw4/)

## Homework 5 — Classical ML Baseline (Course Project)

Develop, train and evaluate a classical ML baseline for skin-lesion
classification (`diagnosis_1`: Benign / Indeterminate / Malignant).

- Notebook (main deliverable, with all results and written analysis):
  [notebooks/homework_part_5.ipynb](notebooks/homework_part_5.ipynb)
- Notebook (HTML export): [notebooks/homework_part_5.html](notebooks/homework_part_5.html)
- Feature extraction module: [milk10k_pipeline/features.py](milk10k_pipeline/features.py)
- Baseline module (feature sets, models, CV, metrics, bootstrap CIs, cost rule):
  [milk10k_pipeline/baseline.py](milk10k_pipeline/baseline.py)
- Feature extraction script: [scripts/extract_features.py](scripts/extract_features.py)
- Tests: [tests/test_features.py](tests/test_features.py)
- Figures: [reports/figures_hw5/](reports/figures_hw5/)
- Result tables (CSV): [reports/hw5_results/](reports/hw5_results/)

**How to reproduce.**

```bash
python scripts/extract_features.py      # ~10 min, writes data/processed/image_features.csv.gz
jupyter nbconvert --to notebook --execute --inplace notebooks/homework_part_5.ipynb
pytest tests/test_features.py
```

The notebook re-uses `reports/hw5_results/cv_grid.csv` and `tuning.csv` if
they exist (the full 50-configuration grid and the grid search take about
1.5 h on a laptop); delete them to recompute from scratch.

## Homework 6 — MLP, `nn.Conv2d`, and raw pixels

1. MLP on the Session 5 features, added to the scoreboard next to the baseline.
2. `nn.Conv2d` arguments for Session 4's kernel size, stride and padding.
3. MLP on raw pixels (32×32×3 = 3,072 numbers) vs the feature baseline, and why.

- Notebook (all answers): [notebooks/homework_part_6.ipynb](notebooks/homework_part_6.ipynb)
- Notebook (PDF, submitted): [notebooks/homework_part_6.pdf](notebooks/homework_part_6.pdf)
- MLP model, training loop, 32×32 pixel loader: [milk10k_pipeline/mlp.py](milk10k_pipeline/mlp.py)
- Tests: [tests/test_mlp.py](tests/test_mlp.py)
- Project scoreboard (all sessions, same splits and metrics): [reports/scoreboard.csv](reports/scoreboard.csv)
- Figures: [reports/figures_hw6/](reports/figures_hw6/)
- Result tables (CSV): [reports/hw6_results/](reports/hw6_results/)

Reproduce (about 20 min on a laptop CPU; needs `data/processed/image_features.csv.gz`
from `scripts/extract_features.py`; the 32×32 pixel cache is built on first run):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/homework_part_6.ipynb
pytest tests/test_mlp.py
```
