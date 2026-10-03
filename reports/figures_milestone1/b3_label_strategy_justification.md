# B3 label strategy — justification

## Primary target: keep all three diagnosis_1 classes

`diagnosis_1` has three classes with the following lesion counts:

- **Malignant:** 3,634 (69.3%)
- **Benign:** 1,483 (28.4%)
- **Indeterminate:** 123 (2.3%)

We keep `Indeterminate` as its own class rather than merging it into
Malignant or Benign. Two reasons. First, it is a real decision boundary
in the clinical workflow: an "indeterminate" lesion is referred for
biopsy or follow-up, which is a different action from "reassure the
patient" (Benign) or "act now" (Malignant). Merging it into one of the
other two would produce a model whose outputs cannot be mapped to an
action. Second, 123 lesions is small but not negligible: enough to train
a class with class weights, though not enough to expect good per-class
metrics from a single split. We therefore commit to reporting per-class
metrics (macro-F1, balanced accuracy) and to computing class weights on
the train split only.

## Stretch goal: merge rare classes into "other"

The 11-class scheme has a long tail. Five classes have fewer than 50
lesions:

- MAL_OTH: 9
- BEN_OTH: 44
- VASC: 47
- INF: 50
- DF: 52

We merge these five classes into a single `other` bucket for training
and evaluation of the stretch task. The trade-off:

- **Merging loses specificity.** A model trained on `other` cannot tell
  MAL_OTH from DF. For a research paper this would be a problem.
- **Merging is the only option that supports reliable evaluation.** A
  class with 9 lesions cannot be measured on a 70/15/15 split: as shown
  in A3.2, MAL_OTH ends up in both val and test in only 2 out of 10
  seeds. Any per-class metric for such a class would be dominated by
  noise.

The pipeline therefore supports both the raw 11-class label (for EDA
and for downstream analyses where specificity matters) and the merged
6-class stretch label (`BCC`, `NV`, `BKL`, `SCCKA`, `MEL`, `AKIEC`,
`other`), which is what the model will actually train on. The exact
mapping is written to `configs/label_map.json` and loaded by every
later milestone.
