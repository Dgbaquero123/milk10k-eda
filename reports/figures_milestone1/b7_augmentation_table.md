# B7 augmentation table

| augmentation | parameters | medical justification |
|---|---|---|
| RandomHorizontalFlip | p = 0.5 | Skin lesions have no canonical left-right orientation. Flipping does not change the diagnosis. |
| RandomVerticalFlip | p = 0.5 | Same as above: no canonical up-down orientation for a skin close-up. |
| RandomRotation | ± 15 degrees | Small rotations keep the whole lesion inside the frame. Larger rotations would need padding or would push the lesion out. |
| ColorJitter | brightness 0.1, contrast 0.1, saturation 0.05, hue 0.02 | Conservative values (from the A3.5 audit). The hue shift stays under 7 degrees, well inside the Benign-vs-Malignant class gap (42 degrees). Avoids erasing diagnostically meaningful erythema and pigmentation. |
| RandomResizedCrop | scale in (0.85, 1.0) | The crop never removes the lesion; the minimum scale keeps the whole lesion inside the frame in the vast majority of cases. |
| Resize | 224 x 224 | Standard ImageNet input size; uniform across all images so the model learns a single representation. |
| ToTensor + Normalize | mean and std from `configs/norm_stats.json` | Uses train-only statistics (B6). Normalization stabilizes training and matches what the model will see at inference. |

The corresponding grid is in `b7_augmentation_gallery.png`
(3 classes x 8 columns: original + 7 augmented versions).
