"""
quality_report.py
B4: data-quality report for Milestone 1.

Writes reports/quality_report.md with:
  - missing-value handling decision for every column
  - label-consistency checks (2 images per lesion, one per image_type)
  - shortcut cross-tabs (image_manipulation, image_type vs diagnosis_1)
  - list of columns that must NOT be used as model inputs (label leakage)

Note: uses plain pandas formatting (no tabulate dependency).
"""
from pathlib import Path
import pandas as pd

from . import config

OUT = config.REPORTS_DIR / "quality_report.md"


MISSING_HANDLING = [
    ("age_approx", 0.38, "impute with median computed on the train split only"),
    ("anatom_site_general", 37.33, "keep as explicit 'unknown' category"),
    ("anatom_site_special", 98.03, "drop (too much missing to be informative)"),
    ("diagnosis_3", 1.51, "leave NaN; not used as a model input anyway"),
    ("diagnosis_4", 85.48, "leave NaN; not used as a model input anyway"),
    ("melanocytic", 77.18, "drop (single value, NaN means 'not applicable')"),
    ("attribution", 0.0, "drop (constant, uninformative)"),
    ("copyright_license", 0.0, "drop (constant, uninformative)"),
]


def label_consistency(df):
    """Exactly 2 images per lesion, one dermoscopic and one clinical."""
    per_lesion = df.groupby("lesion_id").size()
    n_two = (per_lesion == 2).sum()
    n_not_two = (per_lesion != 2).sum()

    per_type = (df.groupby(["lesion_id", "image_type"]).size()
                  .unstack(fill_value=0))
    both_views = 0
    if "dermoscopic" in per_type.columns and "clinical: close-up" in per_type.columns:
        both_views = ((per_type["dermoscopic"] == 1) &
                      (per_type["clinical: close-up"] == 1)).sum()

    return {
        "n_lesions": int(df["lesion_id"].nunique()),
        "n_lesions_with_two_images": int(n_two),
        "n_lesions_with_wrong_count": int(n_not_two),
        "n_lesions_with_both_views": int(both_views),
    }


def shortcut_tables(df):
    manip = (pd.crosstab(df["image_manipulation"].fillna("none"),
                         df["diagnosis_1"], normalize="index")
               .round(4) * 100)
    itype = (pd.crosstab(df["image_type"],
                         df["diagnosis_1"], normalize="index")
               .round(4) * 100)
    return manip, itype


def _md_table(df, index_name=""):
    """Minimal Markdown table formatter, no external dependency."""
    cols = [index_name] + [str(c) for c in df.columns] if index_name else [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(df, out_path=None):
    out_path = Path(out_path) if out_path else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checks = label_consistency(df)
    manip, itype = shortcut_tables(df)

    lines = []
    lines.append("# Data-quality report — MILK10k Milestone 1\n\n")
    lines.append(f"Generated on {config.SPLIT_DATE} with seed {config.SEED}.\n\n")

    lines.append("## 1. Missing-value handling decisions\n\n")
    lines.append("| column | % missing | decision |\n")
    lines.append("|---|---|---|\n")
    for col, pct, decision in MISSING_HANDLING:
        lines.append(f"| `{col}` | {pct:.2f} | {decision} |\n")
    lines.append("\n")

    lines.append("## 2. Label-consistency checks\n\n")
    lines.append(f"- Unique lesions in the metadata: **{checks['n_lesions']}**\n")
    lines.append(f"- Lesions with exactly 2 images: **{checks['n_lesions_with_two_images']}**\n")
    lines.append(f"- Lesions with a different image count: **{checks['n_lesions_with_wrong_count']}**\n")
    lines.append(f"- Lesions with one dermoscopic and one clinical image: "
                 f"**{checks['n_lesions_with_both_views']}**\n\n")

    lines.append("## 3. Suspicious shortcuts\n\n")
    lines.append("Cross-tab of `image_manipulation` vs `diagnosis_1` "
                 "(rows sum to 100%):\n\n")
    lines.append(_md_table(manip, index_name="image_manipulation"))
    lines.append("\n\n")
    lines.append("Cross-tab of `image_type` vs `diagnosis_1` "
                 "(rows sum to 100%):\n\n")
    lines.append(_md_table(itype, index_name="image_type"))
    lines.append("\n\n")
    lines.append(
        "**Conclusion.** `image_manipulation` is not uniform across the "
        "three classes: the `altered` category has more Benign and fewer "
        "Malignant than `instrument only`. It is a plausible proxy for "
        "which clinic contributed the image rather than for the diagnosis "
        "itself, so it should not be used as a model input without an "
        "ablation study. `image_type` is perfectly balanced by design "
        "(every lesion has one of each), so it carries no information "
        "about the label on its own.\n\n"
    )

    lines.append("## 4. Columns not used as model inputs (label leakage)\n\n")
    lines.append(
        "- `diagnosis_confirm_type`: encodes how the label was confirmed "
        "(histopathology = almost certainly Malignant; clinical assessment "
        "= mostly Benign). Using it leaks the label.\n"
        "- `diagnosis_2`, `diagnosis_3`, `diagnosis_4`: hierarchical "
        "sub-classifications of the target; Cramér's V = 1.0 with "
        "`diagnosis_1`.\n"
        "- `isic_id`, `lesion_id`: identifiers, not features.\n"
        "- `attribution`, `copyright_license`: constants.\n"
    )

    out_path.write_text("".join(lines))
    return out_path, checks, manip, itype


def run():
    print("=" * 72)
    print("B4  Data-quality report")
    print("=" * 72)
    df = pd.read_csv(config.METADATA_PATH)
    out_path, checks, manip, itype = write_report(df)

    print("label-consistency checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print()
    print("image_manipulation vs diagnosis_1:")
    print(manip)
    print()
    print("image_type vs diagnosis_1:")
    print(itype)
    print()
    print("saved:", out_path)


if __name__ == "__main__":
    run()
