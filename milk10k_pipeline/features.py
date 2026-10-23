"""
features.py
Homework 5: hand-crafted image features for a classical ML baseline.

Every image is resized to 224x224 (same IMAGE_SIZE as Milestone 1) and
described by five families of features. Each family is a dict of
named scalars so that columns stay interpretable and families can be
switched on or off when comparing feature sets.

Families (prefix of the column name)
------------------------------------
  color_   Colour statistics in RGB, HSV and CIE-Lab (mean, std, skew,
           percentiles) plus 8-bin hue / saturation / value histograms,
           and lesion-vs-surrounding-skin colour contrast.
  tex_     Texture: grey-level co-occurrence matrix (GLCM) properties and
           uniform Local Binary Pattern (LBP) histograms at two radii.
  filt_    Filter bank from Homework 4: Gaussian blur residual, Sobel
           gradient magnitude, Laplacian, Gabor (4 orientations x 2
           frequencies) response statistics, and Canny edge density.
  shape_   Shape of the lesion mask (Otsu threshold on the blurred
           grey image, largest central component): area, eccentricity,
           solidity, border irregularity (compactness) and asymmetry,
           i.e. the A and B of the ABCD rule.
  hog_     Histogram of Oriented Gradients on a 64x64 grey image,
           pooled to a small fixed vector.

Metadata (age, sex, anatomical site) is not computed here; it is added
at lesion level in `baseline.py`.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import label, regionprops

SIZE = 224
FAMILIES = ("color", "tex", "filt", "shape", "hog")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rgb(path, size: int = SIZE) -> np.ndarray:
    """Load an image as uint8 RGB resized to (size, size)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Lesion mask (used by colour contrast and shape)
# ---------------------------------------------------------------------------

def lesion_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Rough lesion segmentation: lesions are darker than the surrounding
    skin, so Otsu on a blurred grey image, then keep the connected
    component that is closest to the image centre. Returns a bool mask.
    Falls back to a central disc if the threshold finds nothing usable.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = ndimage.binary_fill_holes(th > 0)
    lab = label(th)
    h, w = gray.shape
    best, best_score = None, np.inf
    for r in regionprops(lab):
        if r.area < 0.01 * h * w:
            continue
        cy, cx = r.centroid
        dist = np.hypot(cy - h / 2, cx - w / 2) / (h / 2)
        # Penalise regions touching the border a lot (vignetting, shadows)
        minr, minc, maxr, maxc = r.bbox
        touches = (minr == 0) + (minc == 0) + (maxr == h) + (maxc == w)
        score = dist + 0.5 * touches
        if score < best_score:
            best, best_score = r.label, score
    if best is None:
        yy, xx = np.ogrid[:h, :w]
        return (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (h / 4) ** 2
    return lab == best


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def _stats(prefix: str, x: np.ndarray) -> dict:
    x = x.ravel().astype(np.float64)
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    return {
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": x.std(),
        f"{prefix}_skew": float(stats.skew(x)) if x.std() > 0 else 0.0,
        f"{prefix}_p10": p10,
        f"{prefix}_p50": p50,
        f"{prefix}_p90": p90,
    }


def color_features(rgb: np.ndarray, mask: np.ndarray) -> dict:
    f = {}
    rgbf = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] *= 2.0          # OpenCV hue is 0..179 -> degrees
    hsv[..., 1:] /= 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0

    for i, c in enumerate("RGB"):
        f.update(_stats(f"color_{c}", rgbf[..., i]))
    for i, c in enumerate(["S", "V"]):
        f.update(_stats(f"color_{c}", hsv[..., i + 1]))
    for i, c in enumerate(["L", "a", "b"]):
        f.update(_stats(f"color_lab{c}", lab[..., i]))

    # Circular hue statistics (hue is an angle)
    ang = np.deg2rad(hsv[..., 0].ravel())
    f["color_H_cos"] = float(np.cos(ang).mean())
    f["color_H_sin"] = float(np.sin(ang).mean())

    # 8-bin histograms of H, S, V (normalised)
    for i, (c, rng) in enumerate([("H", (0, 360)), ("S", (0, 1)), ("V", (0, 1))]):
        h, _ = np.histogram(hsv[..., i], bins=8, range=rng)
        h = h / h.sum()
        for k, v in enumerate(h):
            f[f"color_hist{c}_{k}"] = v

    # Lesion vs surrounding skin contrast in Lab (the "C" of ABCD) and
    # number of distinct colours inside the lesion (colour variegation)
    inside, outside = lab[mask], lab[~mask]
    if len(outside) < 50:
        outside = lab.reshape(-1, 3)
    for i, c in enumerate(["L", "a", "b"]):
        f[f"color_contrast_{c}"] = inside[:, i].mean() - outside[:, i].mean()
        f[f"color_in_std_{c}"] = inside[:, i].std()
    f["color_deltaE"] = float(np.linalg.norm(inside.mean(0) - outside.mean(0)))
    q = (rgb[mask] // 64).astype(np.int32)           # 4 levels per channel
    codes, counts = np.unique(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], return_counts=True)
    f["color_n_colours_5pct"] = int((counts / counts.sum() > 0.05).sum())
    return f


def texture_features(gray: np.ndarray) -> dict:
    f = {}
    g32 = (gray // 8).astype(np.uint8)  # 32 grey levels
    glcm = graycomatrix(g32, distances=[1, 3, 5],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        v = graycoprops(glcm, prop)              # (distances, angles)
        for di, d in enumerate([1, 3, 5]):
            f[f"tex_glcm_{prop}_d{d}"] = float(v[di].mean())
    # Entropy of the GLCM at distance 1
    p = glcm[:, :, 0, :].mean(axis=-1)
    p = p[p > 0]
    f["tex_glcm_entropy_d1"] = float(-(p * np.log2(p)).sum())

    for P, R in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
        h = h / h.sum()
        for k, v in enumerate(h):
            f[f"tex_lbp{P}r{R}_{k}"] = v
    return f


def filter_features(gray: np.ndarray) -> dict:
    """Filter bank from Homework 4, summarised by mean and std."""
    f = {}
    g = gray.astype(np.float32) / 255.0
    for s in (1, 3):
        blur = ndimage.gaussian_filter(g, sigma=s)
        f[f"filt_gauss_resid_s{s}_std"] = float((g - blur).std())
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    f.update({k.replace("filt_sobel", "filt_sobel"): v
              for k, v in _stats("filt_sobel", mag).items()
              if k.endswith(("mean", "std", "p90"))})
    lap = ndimage.laplace(ndimage.gaussian_filter(g, 1.0))
    f["filt_log_absmean"] = float(np.abs(lap).mean())
    f["filt_log_std"] = float(lap.std())
    for freq in (0.1, 0.25):
        for t in range(4):
            theta = t * np.pi / 4
            kern = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta,
                                      lambd=1.0 / freq, gamma=0.5, psi=0)
            resp = np.abs(cv2.filter2D(g, cv2.CV_32F, kern))
            f[f"filt_gabor_f{freq}_t{t}_mean"] = float(resp.mean())
            f[f"filt_gabor_f{freq}_t{t}_std"] = float(resp.std())
    edges = cv2.Canny(gray, 50, 150)
    f["filt_canny_density"] = float((edges > 0).mean())
    return f


def shape_features(mask: np.ndarray) -> dict:
    f = {}
    h, w = mask.shape
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return {k: 0.0 for k in ["shape_area", "shape_eccentricity", "shape_solidity",
                                  "shape_extent", "shape_compactness",
                                  "shape_asym_major", "shape_asym_minor"]}
    r = props[0]
    f["shape_area"] = r.area / (h * w)
    f["shape_eccentricity"] = r.eccentricity
    f["shape_solidity"] = r.solidity
    f["shape_extent"] = r.extent
    # Border irregularity: perimeter^2 / (4 pi area), 1 for a circle
    f["shape_compactness"] = (r.perimeter ** 2) / (4 * np.pi * r.area) if r.area else 0.0
    # Asymmetry: rotate mask so the major axis is horizontal, then flip
    # around each principal axis and measure the non-overlap fraction.
    cy, cx = r.centroid
    angle = -np.degrees(r.orientation) + 90
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    rot = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
    area = rot.sum() or 1
    f["shape_asym_major"] = float((rot ^ rot[::-1, :]).sum() / area)
    f["shape_asym_minor"] = float((rot ^ rot[:, ::-1]).sum() / area)
    return f


def hog_features(gray: np.ndarray) -> dict:
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    v = hog(small, orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), feature_vector=True)
    # 3x3 blocks x 4 cells x 9 orientations = 324 values
    return {f"hog_{i}": float(x) for i, x in enumerate(v)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def image_features(path, families=FAMILIES) -> dict:
    """All requested feature families for one image, as a flat dict."""
    rgb = load_rgb(path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = lesion_mask(rgb)
    out = {}
    if "color" in families:
        out.update(color_features(rgb, mask))
    if "tex" in families:
        out.update(texture_features(gray))
    if "filt" in families:
        out.update(filter_features(gray))
    if "shape" in families:
        out.update(shape_features(mask))
    if "hog" in families:
        out.update(hog_features(gray))
    return out
