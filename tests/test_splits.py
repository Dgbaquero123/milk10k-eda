"""
tests/test_splits.py
Property tests for milk10k_pipeline.splits.
"""
import pytest
import pandas as pd
from milk10k_pipeline import config
from milk10k_pipeline.splits import load_lesions_table, split_lesions


@pytest.fixture(scope="module")
def lesions():
    return load_lesions_table()


def test_no_overlap(lesions):
    for seed in [0, 1, 2, 3, 4]:
        tr, va, te = split_lesions(lesions, 0.15, 0.15, seed=seed)
        assert set(tr).isdisjoint(set(va))
        assert set(tr).isdisjoint(set(te))
        assert set(va).isdisjoint(set(te))


def test_sizes_within_tolerance(lesions):
    n = len(lesions)
    for seed in [0, 1, 2, 3, 4]:
        tr, va, te = split_lesions(lesions, 0.15, 0.15, seed=seed)
        assert abs(len(tr) / n - 0.70) <= 0.01 + 1e-9
        assert abs(len(va) / n - 0.15) <= 0.01 + 1e-9
        assert abs(len(te) / n - 0.15) <= 0.01 + 1e-9


def test_reproducible(lesions):
    tr1, va1, te1 = split_lesions(lesions, 0.15, 0.15, seed=42)
    tr2, va2, te2 = split_lesions(lesions, 0.15, 0.15, seed=42)
    assert tr1 == tr2 and va1 == va2 and te1 == te2


def test_all_lesions_accounted_for(lesions):
    tr, va, te = split_lesions(lesions, 0.15, 0.15, seed=0)
    assert set(tr) | set(va) | set(te) == set(lesions["lesion_id"])
    assert len(tr) + len(va) + len(te) == len(lesions)
