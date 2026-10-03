# Data-quality report — MILK10k Milestone 1

Generated on 2026-10-03 with seed 0.

## 1. Missing-value handling decisions

| column | % missing | decision |
|---|---|---|
| `age_approx` | 0.38 | impute with median computed on the train split only |
| `anatom_site_general` | 37.33 | keep as explicit 'unknown' category |
| `anatom_site_special` | 98.03 | drop (too much missing to be informative) |
| `diagnosis_3` | 1.51 | leave NaN; not used as a model input anyway |
| `diagnosis_4` | 85.48 | leave NaN; not used as a model input anyway |
| `melanocytic` | 77.18 | drop (single value, NaN means 'not applicable') |
| `attribution` | 0.00 | drop (constant, uninformative) |
| `copyright_license` | 0.00 | drop (constant, uninformative) |

## 2. Label-consistency checks

- Unique lesions in the metadata: **5240**
- Lesions with exactly 2 images: **5240**
- Lesions with a different image count: **0**
- Lesions with one dermoscopic and one clinical image: **5240**

## 3. Suspicious shortcuts

Cross-tab of `image_manipulation` vs `diagnosis_1` (rows sum to 100%):

| image_manipulation | Benign | Indeterminate | Malignant |
|---|---|---|---|
| altered | 39.70 | 12.84 | 47.46 |
| instrument only | 27.93 | 2.00 | 70.07 |

Cross-tab of `image_type` vs `diagnosis_1` (rows sum to 100%):

| image_type | Benign | Indeterminate | Malignant |
|---|---|---|---|
| clinical: close-up | 28.30 | 2.35 | 69.35 |
| dermoscopic | 28.30 | 2.35 | 69.35 |

**Conclusion.** `image_manipulation` is not uniform across the three classes: the `altered` category has more Benign and fewer Malignant than `instrument only`. It is a plausible proxy for which clinic contributed the image rather than for the diagnosis itself, so it should not be used as a model input without an ablation study. `image_type` is perfectly balanced by design (every lesion has one of each), so it carries no information about the label on its own.

## 4. Columns not used as model inputs (label leakage)

- `diagnosis_confirm_type`: encodes how the label was confirmed (histopathology = almost certainly Malignant; clinical assessment = mostly Benign). Using it leaks the label.
- `diagnosis_2`, `diagnosis_3`, `diagnosis_4`: hierarchical sub-classifications of the target; Cramér's V = 1.0 with `diagnosis_1`.
- `isic_id`, `lesion_id`: identifiers, not features.
- `attribution`, `copyright_license`: constants.
