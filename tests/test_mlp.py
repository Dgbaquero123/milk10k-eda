"""Tests for the Homework 6 MLP module."""
import numpy as np
import pandas as pd
import torch

from milk10k_pipeline import mlp


def test_pixel_matrix_layout_is_channel_first():
    img = np.zeros((32, 32, 3), np.uint8)
    img[..., 0] = 255                                   # pure red
    les = pd.DataFrame({"derm_id": ["a"], "clinical_id": ["b"]})
    X = mlp.pixel_matrix(les, {"a": img, "b": img}, views=("derm", "clin"))
    assert X.shape == (1, 2 * 3072)
    assert X[0, :1024].min() == 1.0 and X[0, 1024:3072].max() == 0.0


def test_channel_standardizer_uses_train_stats_per_channel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3 * 16)).astype(np.float32)
    X[:, :16] += 5
    st = mlp.ChannelStandardizer(3).fit(X)
    Z = st.transform(X).reshape(50, 3, 16)
    assert np.allclose(Z.mean(axis=(0, 2)), 0, atol=1e-5)
    assert np.allclose(Z.std(axis=(0, 2)), 1, atol=1e-4)


def test_mlp_shapes_and_param_count():
    m = mlp.MLP(3072, hidden=(128,), p_drop=0.5)
    assert m(torch.zeros(4, 3072)).shape == (4, 3)
    assert mlp.n_params(m) == 3072 * 128 + 128 + 128 * 3 + 3


def test_training_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 60)
    X = (rng.normal(size=(180, 5)) + 3 * np.eye(3, 5)[y]).astype(np.float32)
    r = mlp.train_mlp(X, y, mlp.TrainConfig(hidden=(16,), max_epochs=200, patience=30))
    assert r.inner_val_f1 > 0.9
    p = mlp.predict_proba(r.model, X)
    assert np.allclose(p.sum(1), 1, atol=1e-5)


def test_class_weights_formula():
    w = mlp.class_weights_from(np.array([0, 0, 0, 1, 2, 2]))
    assert torch.allclose(w, torch.tensor([6 / 9, 6 / 3, 6 / 6]))
