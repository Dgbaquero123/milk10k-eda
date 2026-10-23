"""
baseline.py
Homework 5: classical ML baseline for MILK10k (3 classes, lesion level).

Building blocks used by `notebooks/homework_part_5.ipynb`:

  load_lesion_table()     lesion-level feature matrix built from the cached
                          image features (one dermoscopic + one clinical
                          image per lesion) plus metadata, joined to the
                          Milestone 1 splits.
  feature_columns()       column lists for each feature set.
  make_models()           the candidate classifiers, each wrapped in a
                          leak-free Pipeline (impute -> scale -> model).
  cross_validate_config() 5-fold StratifiedGroupKFold on the TRAIN split.
  evaluate()              metrics on a held-out split.
  bootstrap_ci()          95% bootstrap confidence intervals.
  cost_optimal_predict()  Bayes decision rule under the A3.4 cost matrix.

Rules followed (from Sessions 1-3 and Milestone 1):
  - Unit of evaluation is the LESION; splits are the committed
    lesion-level splits (`splits/*.csv`), so there is no sibling leak.
  - Every statistic that is learned (imputation values, scaling,
    one-hot categories, model parameters) is fitted inside a Pipeline
    on training folds only.
  - Model selection uses cross-validation on train; val is used once
    to confirm the choice; test is used once at the very end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from . import config

CLASSES = ["Benign", "Indeterminate", "Malignant"]
FEATURES_PATH = config.ROOT / "data" / "processed" / "image_features.csv.gz"

# A3.4 cost matrix, rows = truth, cols = prediction
COST = pd.DataFrame([[0, 5, 1],
                     [5, 0, 5],
                     [50, 5, 0]], index=CLASSES, columns=CLASSES)

IMAGE_FAMILIES = ["color", "tex", "filt", "shape", "hog"]
META_NUM = ["age"]
META_CAT = ["sex", "site"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_lesion_table(features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """
    One row per lesion with columns:
      lesion_id, split, diagnosis_1, dx, age, sex, site,
      derm__<feature> ..., clin__<feature> ...
    """
    feats = pd.read_csv(features_path)
    parts = []
    for split in ["train", "val", "test"]:
        s = pd.read_csv(config.SPLITS_DIR / f"{split}.csv")
        s["split"] = split
        parts.append(s)
    les = pd.concat(parts, ignore_index=True)

    derm = feats.add_prefix("derm__").rename(columns={"derm__isic_id": "derm_id"})
    clin = feats.add_prefix("clin__").rename(columns={"clin__isic_id": "clinical_id"})
    out = les.merge(derm, on="derm_id", how="left").merge(clin, on="clinical_id", how="left")
    assert len(out) == len(les) == 5240, "every lesion must appear exactly once"
    assert out.filter(like="derm__").notna().all().all(), "missing image features"
    assert out["lesion_id"].is_unique
    return out


def feature_columns(df: pd.DataFrame, families, views=("derm", "clin"),
                    meta: bool = False) -> list[str]:
    """Columns for a feature set: image families x views (+ metadata)."""
    cols = []
    for v in views:
        for fam in families:
            cols += [c for c in df.columns if c.startswith(f"{v}__{fam}_")]
    if meta:
        cols += META_NUM + META_CAT
    return cols


# The feature sets compared in Step 2 (name -> (families, views, meta))
FEATURE_SETS = {
    "meta only":                  ([], ("derm", "clin"), True),
    "colour":                     (["color"], ("derm", "clin"), False),
    "texture (GLCM+LBP)":         (["tex"], ("derm", "clin"), False),
    "filter bank (HW4)":          (["filt"], ("derm", "clin"), False),
    "shape (ABCD)":               (["shape"], ("derm", "clin"), False),
    "HOG":                        (["hog"], ("derm", "clin"), False),
    "hand-crafted (C+T+F+S)":     (["color", "tex", "filt", "shape"], ("derm", "clin"), False),
    "hand-crafted + HOG":         (["color", "tex", "filt", "shape", "hog"], ("derm", "clin"), False),
    "hand-crafted + meta":        (["color", "tex", "filt", "shape"], ("derm", "clin"), True),
    "all (hand-crafted+HOG+meta)": (["color", "tex", "filt", "shape", "hog"], ("derm", "clin"), True),
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _preprocessor(cols: list[str]) -> ColumnTransformer:
    num = [c for c in cols if c not in META_CAT]
    cat = [c for c in cols if c in META_CAT]
    transformers = [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                      ("scale", StandardScaler())]), num)]
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(transformers)


def make_models(seed: int = config.SEED) -> dict:
    """Candidate classifiers. All use class balancing for the 2.5:1:28 imbalance."""
    return {
        "LogReg": LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
        "SVM-RBF": SVC(C=1.0, gamma="scale", class_weight="balanced",
                       probability=True, random_state=seed),
        "kNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                               class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=seed),
        "HistGB": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                 early_stopping=True,
                                                 validation_fraction=0.15,
                                                 n_iter_no_change=20,
                                                 class_weight="balanced",
                                                 random_state=seed),
    }


def make_pipeline(model, cols: list[str]) -> Pipeline:
    return Pipeline([("prep", _preprocessor(cols)), ("model", clone(model))])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_cost(y_true, y_pred, cost: pd.DataFrame = COST) -> float:
    return float(np.mean([cost.loc[t, p] for t, p in zip(y_true, y_pred)]))


def metrics(y_true, y_pred, proba=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rec = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    f1c = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    out = {
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_Benign": rec[0],
        "recall_Indeterminate": rec[1],
        "recall_Malignant": rec[2],
        "f1_Benign": f1c[0],
        "f1_Indeterminate": f1c[1],
        "f1_Malignant": f1c[2],
        "cost_per_lesion": expected_cost(y_true, y_pred),
    }
    if proba is not None:
        out["macro_auc"] = roc_auc_score(y_true, proba, labels=CLASSES,
                                         multi_class="ovr", average="macro")
    return out


def cost_optimal_predict(proba: np.ndarray, cost: pd.DataFrame = COST) -> np.ndarray:
    """Bayes rule: pick the prediction with the lowest expected cost."""
    exp_cost = proba @ cost.loc[CLASSES, CLASSES].values   # (n, n_pred)
    return np.array(CLASSES)[exp_cost.argmin(axis=1)]


# ---------------------------------------------------------------------------
# Cross-validation and evaluation
# ---------------------------------------------------------------------------

def threshold_predict(proba: np.ndarray, t_malignant: float) -> np.ndarray:
    """
    Screening rule: call Malignant whenever P(Malignant) >= t, otherwise
    the more likely of Benign / Indeterminate.
    """
    rest = np.array(CLASSES[:2])[proba[:, :2].argmax(axis=1)]
    return np.where(proba[:, 2] >= t_malignant, "Malignant", rest)


def threshold_for_sensitivity(y_true, p_malignant, target: float = 0.95) -> float:
    """Largest threshold on P(Malignant) whose Malignant recall >= target."""
    y = np.asarray(y_true) == "Malignant"
    for t in np.sort(np.unique(p_malignant))[::-1]:
        if (p_malignant[y] >= t).mean() >= target:
            return float(t)
    return 0.0


def cross_validate_config(train: pd.DataFrame, cols: list[str], model,
                          n_splits: int = 5, seed: int = config.SEED) -> dict:
    """
    5-fold StratifiedGroupKFold on train (grouped by lesion_id, stratified
    on diagnosis_1). Returns mean and std of every metric plus the
    out-of-fold probabilities.
    """
    X, y, groups = train[cols], train["diagnosis_1"].values, train["lesion_id"].values
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(train), len(CLASSES)))
    fold_scores = []
    for tr, va in cv.split(X, y, groups):
        pipe = make_pipeline(model, cols).fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[va])
        oof[va] = p
        fold_scores.append(metrics(y[va], pipe.predict(X.iloc[va]), p))
    fs = pd.DataFrame(fold_scores)
    res = {f"{k}_mean": v for k, v in fs.mean().items()}
    res.update({f"{k}_std": v for k, v in fs.std().items()})
    res["oof_proba"] = oof
    return res


def evaluate(pipe: Pipeline, df: pd.DataFrame, cols: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    proba = pipe.predict_proba(df[cols])
    pred = pipe.predict(df[cols])
    return metrics(df["diagnosis_1"].values, pred, proba), pred, proba


def bootstrap_ci(y_true, y_pred, proba, n_boot: int = 1000,
                 seed: int = config.SEED) -> pd.DataFrame:
    """95% percentile bootstrap CI for every metric (resampling lesions)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y_true[idx])) < len(CLASSES):
            continue
        rows.append(metrics(y_true[idx], y_pred[idx], proba[idx]))
    b = pd.DataFrame(rows)
    point = metrics(y_true, y_pred, proba)
    return pd.DataFrame({"value": point,
                         "ci_low": b.quantile(0.025),
                         "ci_high": b.quantile(0.975)})


def confusion(y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES),
                        index=[f"true {c}" for c in CLASSES],
                        columns=[f"pred {c}" for c in CLASSES])
