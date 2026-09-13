# MILK10k — Exploratory Data Analysis

Course project (Computer Vision & Speech Recognition, EADA).
Dataset: MILK10k — 10,480 images, 5,240 lesions.

## Clinical Task

Skin cancer is one of the most common cancers worldwide. Dermatologists examine
skin lesions using both clinical close-up photographs and dermoscopic images.
MILK10k pairs both views for each of 5,240 lesions with metadata (age, sex,
anatomic site, skin tone, diagnosis, ground-truth method).

Primary goal: classify each lesion as Benign / Malignant / Indeterminate.
Stretch goal: full 11-class diagnostic scheme.

## Structure

- data/raw/milk10k/          (dataset, gitignored)
- notebooks/01_milk10k_eda.ipynb
- src/
- reports/figures/
