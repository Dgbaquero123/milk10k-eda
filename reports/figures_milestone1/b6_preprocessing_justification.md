# B6 preprocessing decisions — justification

## Input resolution: 224 × 224

The B1 integrity check confirmed that **every one of the 10,480 images
in MILK10k is 600 × 450 pixels**, so there is no image larger than the
target resolution and no aspect-ratio diversity to worry about. 224 × 224
is the standard input size for ImageNet-pretrained CNNs (ResNet, ViT,
EfficientNet), so choosing it means the model can reuse pretrained
weights without any modification. 256 × 256 would also work and would
preserve slightly more detail, but at the cost of ~30% more compute per
image and no compatibility benefit. We choose 224.

The 600 × 450 aspect ratio is 4:3, so resizing to 224 × 224 squashes the
image slightly along the horizontal axis. The distortion is uniform
across all images, so the model can learn it.

## Combining the two views

Each lesion has one dermoscopic and one clinical close-up image. Three
options:

- **(a) independent images with the lesion label**: 2× effective
  samples per lesion, but breaks the lesion-level unit. If the two
  images of a lesion end up in different splits, the metrics are
  contaminated by leakage; even if they don't, evaluating per image
  double-counts each lesion and over-weights the majority class. Not
  chosen.
- **(b) one image type only**: simple and leak-free, but throws away
  half the dataset and the complementary information from the other
  view. Not chosen.
- **(c) both views per lesion, one tensor each**: uses all the data,
  matches the lesion-level unit, and lets the model learn to combine
  the two views (early or late fusion). Chosen.

With option (c), the evaluation unit is the lesion. `aggregate_predictions`
(implemented in A3.6) averages the two image-level class probability
vectors into one per-lesion prediction. Reporting per-image metrics
would inflate the effective sample size and mask the leakage risk.

## Normalization

Per-channel mean and std are computed on the **train split only** (3,668
lesions × 2 views = 7,336 images). Applying the same statistics to
validation and test would leak test information into the training
pipeline. The values are saved to `configs/norm_stats.json` and are
loaded by every later stage (training, inference, and any transfer
learning setup). ImageNet mean/std are *not* used because MILK10k is
dermoscopic skin, which is far from natural photos in colour
distribution; the ImageNet statistics would be a poor match.
