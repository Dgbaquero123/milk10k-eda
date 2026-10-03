# B2 written analysis — Why MILK10k does not reflect real-world prevalence

MILK10k is not a random sample of skin lesions seen in clinics. Of the
5,240 lesions, 69.3% are labelled Malignant, 28.4% Benign, and 2.3%
Indeterminate. In a real primary-care setting, malignant skin lesions are
a small minority of everything a dermatologist looks at; most lesions are
benign nevi, seborrheic keratoses, or other harmless findings. The
difference is not a data error but a **sampling bias by construction**:
the dataset is biopsy-enriched. 95.7% of the images have
`diagnosis_confirm_type == "histopathology"`, which means the lesion was
sent for a biopsy and confirmed under the microscope. Clinicians biopsy
lesions they are worried about, so malignant and borderline cases are
systematically over-represented relative to the general population.

The direct consequence for the pipeline is that the model trained on
MILK10k will be **evaluated on a distribution that is very different
from the one it would face in deployment**. A classifier that reaches
85% accuracy on this dataset is not 85% accurate in a clinic; it is
accurate on a population where 69% of the lesions are malignant and
almost all of them went to biopsy. This has three practical implications
for Milestone 2 and beyond:

1. **The reported accuracy is not a prevalence estimate.** It should
   always be paired with a description of the population it was measured
   on.
2. **Class-weighted loss functions and stratified splits are mandatory,
   not optional.** Without them the model will simply learn the biopsy
   enrichment.
3. **A clinician using the model should be told explicitly** that it has
   been trained on biopsy-confirmed lesions, and that it does not tell
   them how worried they should be about a lesion that has not yet been
   flagged for biopsy.
