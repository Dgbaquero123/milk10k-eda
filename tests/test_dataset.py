"""
tests/test_dataset.py
Smoke tests for milk10k_pipeline.dataset.
"""
import pytest
import torch
from milk10k_pipeline import config
from milk10k_pipeline.dataset import build_loaders


@pytest.fixture(scope="module")
def loaders():
    return build_loaders(batch_size=8, num_workers=0)


def test_train_loader_returns_tensors(loaders):
    train_loader, _, _, _ = loaders
    batch = next(iter(train_loader))
    assert batch is not None
    assert torch.is_tensor(batch["derm"])
    assert torch.is_tensor(batch["clinical"])
    assert batch["derm"].shape == (8, 3, 224, 224)
    assert batch["clinical"].shape == (8, 3, 224, 224)
    assert batch["label"].dtype == torch.int64
    assert len(batch["lesion_id"]) == 8


def test_val_loader_is_deterministic(loaders):
    _, val_loader, _, _ = loaders
    b1 = next(iter(val_loader))
    b2 = next(iter(val_loader))
    assert torch.equal(b1["derm"], b2["derm"])
    assert torch.equal(b1["clinical"], b2["clinical"])


def test_train_and_val_have_no_lesion_overlap(loaders):
    train_loader, val_loader, _, _ = loaders
    tr_ids = set()
    for b in train_loader:
        if b is None:
            continue
        tr_ids.update(b["lesion_id"])
    va_ids = set()
    for b in val_loader:
        if b is None:
            continue
        va_ids.update(b["lesion_id"])
    assert tr_ids.isdisjoint(va_ids)


def test_missing_file_raises():
    import pandas as pd
    from milk10k_pipeline.dataset import LesionDataset
    df = pd.DataFrame([{
        "lesion_id": "FAKE",
        "derm_id": "NOT_A_REAL_IMAGE",
        "clinical_id": "NOT_A_REAL_IMAGE",
        "diagnosis_1": "Benign",
        "dx": "NV",
    }])
    ds = LesionDataset(df, allow_missing=False)
    with pytest.raises(FileNotFoundError):
        _ = ds[0]
