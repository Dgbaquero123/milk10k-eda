"""
label_eda.py
B2: label-centred EDA for Milestone 1.

Produces:
  - class distribution figures (diagnosis_1 and 11-class, log scale)
  - the mapping between diagnosis_1 and the 11-class scheme
  - a 3x4 gallery with one example per 11-class label
  - a summary table "Session 2 finding -> still true -> consequence"
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from . import config

OUT = config.REPORTS_DIR / "figures_milestone1"


def _load_gt_long():
    gt = pd.read_csv(config.GT_PATH)
    class_cols = [c for c in gt.columns if c != "lesion_id"]
    long = gt.melt(id_vars="lesion_id", value_vars=class_cols,
                   var_name="dx", value_name="is_dx")
    return long[long["is_dx"] == 1][["lesion_id", "dx"]]


def class_distributions(out_dir=OUT):
    """Bar charts: diagnosis_1 (linear) and 11-class (log)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.METADATA_PATH)
    gt_long = _load_gt_long()

    # diagnosis_1 per lesion
    dx1 = (df.drop_duplicates("lesion_id")["diagnosis_1"]
             .value_counts()
             .sort_values(ascending=False))

    # 11-class per lesion
    dx11 = gt_long["dx"].value_counts().sort_values(ascending=False)

    # Figure 1: diagnosis_1
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(dx1.index, dx1.values, color="steelblue")
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("# lesions")
    ax.set_title("MILK10k — diagnosis_1 distribution (5,240 lesions)")
    plt.tight_layout()
    fig.savefig(out_dir / "b2_diagnosis_1_distribution.png", dpi=120)
    plt.close(fig)

    # Figure 2: 11-class log scale
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(dx11.index, dx11.values, color="tab:orange")
    ax.bar_label(bars, fmt="%d", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("# lesions (log scale)")
    ax.set_title("MILK10k — 11-class distribution (5,240 lesions)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "b2_11class_distribution_log.png", dpi=120)
    plt.close(fig)

    # Save the tables
    dx1.rename("n_lesions").to_frame().to_csv(out_dir / "b2_diagnosis_1_counts.csv")
    dx11.rename("n_lesions").to_frame().to_csv(out_dir / "b2_11class_counts.csv")

    return dx1, dx11


def gallery_3x4(out_dir=OUT, ncols=4):
    """3x4 gallery with one example per 11-class label."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.METADATA_PATH)
    gt_long = _load_gt_long()
    merged = df.merge(gt_long, on="lesion_id", how="inner")

    # Pick one example per class (deterministic)
    picks = (merged.sort_values("isic_id")
                    .groupby("dx", as_index=False)
                    .first())

    classes = list(picks["dx"])
    fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(picks):
            ax.axis("off")
            continue
        row = picks.iloc[i]
        path = config.image_path(row["isic_id"])
        if not path.exists():
            ax.axis("off")
            continue
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{row['dx']}\n{row['diagnosis_1']}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "b2_gallery_3x4.png", dpi=120)
    plt.close(fig)
    return picks


def session2_findings_table(out_dir=OUT):
    """A table of Session 2 findings and their consequences for M1."""
    rows = [
        {
            "finding": "age_approx is strongly associated with diagnosis_1 "
                       "(Kruskal-Wallis p ≈ 2.4e-255).",
            "still_true_full_dataset": "Yes — the full dataset has the same "
                                       "10480 rows as Session 2.",
            "consequence_for_pipeline": "age_approx can be used as a model "
                                        "input after median imputation.",
        },
        {
            "finding": "concomitant_biopsy and anatom_site_special are strongly "
                       "associated with the target.",
            "still_true_full_dataset": "Yes, but anatom_site_special is 98% "
                                       "missing; concomitant_biopsy has 0% missing.",
            "consequence_for_pipeline": "Drop anatom_site_special; keep "
                                        "concomitant_biopsy as a candidate feature.",
        },
        {
            "finding": "diagnosis_2, diagnosis_3, diagnosis_4 leak the target "
                       "hierarchically (Cramér's V = 1.0).",
            "still_true_full_dataset": "Yes.",
            "consequence_for_pipeline": "Do not use any diagnosis_2/3/4 as "
                                        "model inputs. They are label leakage.",
        },
        {
            "finding": "Color statistics overlap heavily across classes; "
                       "file size is not a usable shortcut (AUC ≈ 0.5).",
            "still_true_full_dataset": "Yes (confirmed in A2.2).",
            "consequence_for_pipeline": "Rely on learned CNN features and "
                                        "texture, not on hand-crafted colour "
                                        "statistics.",
        },
        {
            "finding": "MILK10k is biopsy-enriched (69% Malignant).",
            "still_true_full_dataset": "Yes (full metadata).",
            "consequence_for_pipeline": "Use class weights and stratified "
                                        "splits; report balanced accuracy and "
                                        "macro-F1, not accuracy.",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "b2_session2_findings.csv", index=False)
    return table


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    print("B2  Label-centred EDA")
    print("output:", OUT)
    dx1, dx11 = class_distributions()
    print("diagnosis_1 counts:")
    print(dx1.to_string())
    print()
    print("11-class counts:")
    print(dx11.to_string())
    picks = gallery_3x4()
    print()
    print("gallery picks (one per 11-class label):")
    print(picks[["dx", "diagnosis_1", "isic_id"]].to_string(index=False))
    table = session2_findings_table()
    print()
    print("Session 2 findings -> consequences table:")
    print(table.to_string(index=False))
    print()
    print("B2 done.")


if __name__ == "__main__":
    run()
