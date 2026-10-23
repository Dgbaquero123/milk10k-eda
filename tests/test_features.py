"""Tests for the Homework 5 feature extractor and baseline helpers."""
import numpy as np
import pandas as pd
import pytest

from milk10k_pipeline import features as F
from milk10k_pipeline import baseline as bl


def _synthetic_lesion(size=224):
    """Light skin with a dark disc in the centre."""
    rgb = np.full((size, size, 3), (220, 180, 160), np.uint8)
    yy, xx = np.ogrid[:size, :size]
    rgb[(yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= 40 ** 2] = (90, 50, 40)
    return rgb


def test_mask_finds_central_dark_disc():
    m = F.lesion_mask(_synthetic_lesion())
    area = m.mean()
    assert abs(area - np.pi * 40 ** 2 / 224 ** 2) < 0.01
    assert m[112, 112] and not m[5, 5]


def test_shape_of_a_disc_is_symmetric_and_compact():
    s = F.shape_features(F.lesion_mask(_synthetic_lesion()))
    assert s["shape_asym_major"] < 0.05 and s["shape_asym_minor"] < 0.05
    assert s["shape_eccentricity"] < 0.2
    assert 0.9 < s["shape_compactness"] < 1.3


def test_all_families_are_finite_and_fixed_length(tmp_path):
    import cv2
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), cv2.cvtColor(_synthetic_lesion(300), cv2.COLOR_RGB2BGR))
    f = F.image_features(p)
    assert len(f) == 484
    assert np.isfinite(list(f.values())).all()
    assert {k.split("_")[0] for k in f} == set(F.FAMILIES)


def test_cost_optimal_rule_prefers_malignant_when_uncertain():
    # 10% Malignant is already enough to flag it: 0.1*50 > 0.9*1
    proba = np.array([[0.9, 0.0, 0.1], [0.99, 0.0, 0.01]])
    assert list(bl.cost_optimal_predict(proba)) == ["Malignant", "Benign"]


def test_expected_cost_matches_matrix():
    y = ["Malignant", "Benign", "Indeterminate"]
    p = ["Benign", "Malignant", "Indeterminate"]
    assert bl.expected_cost(y, p) == pytest.approx((50 + 1 + 0) / 3)
