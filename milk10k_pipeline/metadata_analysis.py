"""
metadata_analysis.py
Part 1: metadata analysis vs. target (diagnosis_1).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

TARGET = "diagnosis_1"
ID_COLS = ["isic_id", "lesion_id"]
FREE_TEXT_COLS = ["attribution", "copyright_license"]

CATEGORICAL_COLS = [
    "anatom_site_general",
    "anatom_site_special",
    "concomitant_biopsy",
    "diagnosis_2",
    "diagnosis_3",
    "diagnosis_4",
    "diagnosis_confirm_type",
    "image_manipulation",
    "image_type",
    "melanocytic",
    "sex",
]

NUMERIC_COLS = ["age_approx"]


def load_metadata(path):
    return pd.read_csv(path)


def summarize_columns(df):
    rows = []
    for col in df.columns:
        s = df[col]
        if col in ID_COLS:
            kind = "identifier"
        elif col == TARGET:
            kind = "target"
        elif col in FREE_TEXT_COLS:
            kind = "free text"
        elif col in NUMERIC_COLS:
            kind = "numeric"
        elif s.dtype == bool:
            kind = "boolean"
        elif s.nunique() <= 20:
            kind = "categorical"
        else:
            kind = "free text"
        rows.append({
            "column": col,
            "type": kind,
            "dtype": str(s.dtype),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "n_unique": int(s.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def categorical_vs_target(df, col, figures_dir=None, top_k=8):
    sub = df[[col, TARGET]].dropna()
    if sub.empty:
        return None, None, None, None, None
    top_cats = sub[col].value_counts().head(top_k).index
    sub = sub[sub[col].isin(top_cats)]
    ct = pd.crosstab(sub[col], sub[TARGET])
    ct_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    chi2, p, dof, _ = stats.chi2_contingency(ct.values)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if min(ct.shape) > 1 else np.nan
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_title(f"Distribucion de {TARGET} por {col}")
        ax.set_ylabel("Numero de imagenes")
        ax.set_xlabel(col)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / f"part1_cat_{col}.png", dpi=120)
        plt.close()
    return ct, ct_pct, chi2, p, cramers_v


def numeric_vs_target(df, col, figures_dir=None):
    sub = df[[col, TARGET]].dropna()
    if sub.empty:
        return None, None, None
    groups = [g[col].values for _, g in sub.groupby(TARGET)]
    group_stats = sub.groupby(TARGET)[col].agg(["count", "mean", "median", "std"]).round(2)
    F, p_anova = stats.f_oneway(*groups)
    H, p_kruskal = stats.kruskal(*groups)
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=sub, x=TARGET, y=col, ax=ax)
        ax.set_title(f"{col} por {TARGET}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"part1_num_{col}.png", dpi=120)
        plt.close()
    return group_stats, (H, p_kruskal), (F, p_anova)


def run_part1(metadata_path, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_metadata(metadata_path)
    summary = summarize_columns(df)
    cat_results = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            ct, ct_pct, chi2, p, v = categorical_vs_target(df, col, figures_dir)
            cat_results[col] = {
                "table": ct,
                "table_pct": ct_pct,
                "chi2": chi2,
                "p_value": p,
                "cramers_v": round(v, 4) if v is not None else None,
            }
    num_results = {}
    for col in NUMERIC_COLS:
        if col in df.columns:
            gs, kw, anova = numeric_vs_target(df, col, figures_dir)
            num_results[col] = {"group_stats": gs, "kruskal": kw, "anova": anova}
    return {"df": df, "summary": summary, "categorical": cat_results, "numeric": num_results}
