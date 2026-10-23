"""
generate_hw5_submission.py
Builds reports/hw5_submission.pdf: the Homework 5 hand-in with the links
to the code in the GitHub repository and a one-page summary of results.

Usage (from the repository root, after running the notebook):
    python scripts/generate_hw5_submission.py
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "hw5_results"
FIG = ROOT / "reports" / "figures_hw5"
OUT = ROOT / "reports" / "hw5_submission.pdf"

REPO = "https://github.com/Dgbaquero123/milk10k-eda"
BLOB = f"{REPO}/blob/main"
TREE = f"{REPO}/tree/main"

BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d2")
TINT = colors.HexColor("#eef4fc")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], alignment=TA_LEFT, fontSize=18,
                    leading=22, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10, leading=14, textColor=INK2)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, leading=16,
                    textColor=INK, spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.5, leading=13.5, textColor=INK)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.5, leading=11.5, textColor=INK2)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.5, leading=11)


def link(url, text=None):
    return f'<link href="{url}" color="#2a78d6"><u>{text or url}</u></link>'


def table(data, col_widths, header=True, zebra=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    if header:
        style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK2)]
    t.setStyle(TableStyle(style))
    return t


def fig(name, width_cm):
    p = FIG / name
    img = Image(str(p))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_cm * cm
    img.drawHeight = width_cm * cm * ratio
    return img


def main():
    choice = pd.read_csv(RES / "final_choice.csv", index_col=0).iloc[:, 0]
    tuning = pd.read_csv(RES / "tuning.csv").set_index("model")
    ci_a = pd.read_csv(RES / "test_metrics_argmax_ci.csv", index_col=0)
    ci_s = pd.read_csv(RES / "test_metrics_screening95_ci.csv", index_col=0)
    val = pd.read_csv(RES / "val_candidates.csv").set_index("candidate")
    dummies = pd.read_csv(RES / "dummies.csv").query("split == 'test'").set_index("dummy")
    t95 = float(pd.read_csv(RES / "screening_threshold.csv", index_col=0).iloc[0, 0])

    story = []
    story += [Paragraph("Homework 5 — Classical ML Baseline for Skin-Lesion Classification", H1),
              Paragraph("Daniela Gonzalez · Computer Vision and Speech Recognition — EADA · "
                        "Dataset: MILK10k (5,240 lesions, dermoscopic + clinical image each)", SUB),
              Spacer(1, 6),
              Paragraph(f"<b>Repository:</b> {link(REPO)}", BODY)]

    # --- Links -------------------------------------------------------------
    story.append(Paragraph("Links to the code", H2))
    rows = [["Deliverable", "Link"]]
    items = [
        ("Main notebook (code, results, written analysis)", f"{BLOB}/notebooks/homework_part_5.ipynb"),
        ("Notebook exported to HTML", f"{BLOB}/notebooks/homework_part_5.html"),
        ("Feature extraction module (colour, texture, HW4 filters, shape, HOG)",
         f"{BLOB}/milk10k_pipeline/features.py"),
        ("Baseline module (feature sets, models, CV, metrics, CIs, cost rule)",
         f"{BLOB}/milk10k_pipeline/baseline.py"),
        ("Feature extraction script", f"{BLOB}/scripts/extract_features.py"),
        ("Unit tests for features and decision rules", f"{BLOB}/tests/test_features.py"),
        ("Figures", f"{TREE}/reports/figures_hw5"),
        ("Result tables (CSV)", f"{TREE}/reports/hw5_results"),
        ("Splits reused from Milestone 1 (lesion-level, grouped)", f"{TREE}/splits"),
        ("Submission index for all homeworks", f"{BLOB}/SUBMISSION.md"),
    ]
    for label, url in items:
        rows.append([Paragraph(label, CELL), Paragraph(link(url, url.replace("https://", "")), CELL)])
    story.append(table(rows, [6.2 * cm, 11.0 * cm]))

    # --- Method ------------------------------------------------------------
    story.append(Paragraph("What was done", H2))
    story.append(Paragraph(
        "<b>Task.</b> Predict <i>diagnosis_1</i> (Benign / Indeterminate / Malignant) per lesion. "
        "<b>Features</b> (484 per image, both views concatenated per lesion): colour statistics and "
        "histograms in RGB/HSV/Lab with lesion-vs-skin contrast; GLCM and LBP texture; the Homework 4 "
        "filter bank (Gaussian, Sobel, LoG, Gabor, Canny); ABCD-style shape of an Otsu lesion mask; "
        "HOG; plus metadata (age, sex, site). "
        "<b>Protocol.</b> Milestone 1 lesion-level splits (no sibling leak). 10 feature sets × 5 models "
        "(logistic regression, SVM-RBF, k-NN, random forest, histogram gradient boosting) compared with "
        "5-fold StratifiedGroupKFold on train; the two best models tuned with GridSearchCV in the same CV; "
        "final choice on val; refit on train+val and test used once, with 1,000-sample bootstrap 95% CIs. "
        "All preprocessing inside sklearn Pipelines. Primary metric macro-F1; also balanced accuracy, "
        "macro AUC, per-class recall and the A3.4 clinical cost.", BODY))

    story.append(KeepTogether([
        Paragraph("Model selection (5-fold CV on train)", H2),
        fig("fig2_cv_grid_macro_f1.png", 14.5),
        Paragraph(
        f"Best feature set: <b>{choice['best_feature_set']}</b> ({int(float(choice['n_features']))} features). "
        f"Tuned CV macro-F1: HistGB {tuning.loc['HistGB', 'tuned_cv_macro_f1']:.3f}, "
        f"SVM-RBF {tuning.loc['SVM-RBF', 'tuned_cv_macro_f1']:.3f}. "
        f"Final model chosen on val: <b>{choice['final']}</b> (val macro-F1 "
        f"{val.loc[choice['final'], 'macro_f1']:.3f}). Clinical view alone is clearly weaker "
        "(0.550) than dermoscopic alone (0.586); both views together score 0.601, a gain within CV noise.", SMALL)]))

    story.append(Paragraph("Final results on the test split (786 lesions)", H2))

    def f(ci, k):
        return f"{ci.loc[k, 'value']:.3f} [{ci.loc[k, 'ci_low']:.3f}, {ci.loc[k, 'ci_high']:.3f}]"

    metrics = [("macro-F1", "macro_f1"), ("balanced accuracy", "balanced_acc"),
               ("accuracy", "accuracy"), ("macro AUC (one-vs-rest)", "macro_auc"),
               ("recall Malignant", "recall_Malignant"), ("recall Benign", "recall_Benign"),
               ("recall Indeterminate", "recall_Indeterminate"),
               ("expected cost per lesion (A3.4)", "cost_per_lesion")]
    HB = ParagraphStyle("HB", parent=CELL, fontName="Helvetica-Bold")
    rows = [[Paragraph(h, HB) for h in ["Metric", "Baseline, argmax",
                                        f"Baseline, screening<br/>(t = {t95:.2f})",
                                        "Dummy: always Malignant", "Dummy: stratified random"]]]
    for name, k in metrics:
        d1 = "0.500" if k == "macro_auc" else f"{dummies.loc['always-Malignant', k]:.3f}"
        d2 = "0.500" if k == "macro_auc" else f"{dummies.loc['stratified-random', k]:.3f}"
        rows.append([name, f(ci_a, k), f(ci_s, k), d1, d2])
    t = table(rows, [4.1 * cm, 3.7 * cm, 3.7 * cm, 2.85 * cm, 2.85 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (1, 1), (1, 1), TINT),
                           ("FONT", (1, 1), (1, 1), "Helvetica-Bold", 8.5)]))
    story.append(t)
    story.append(Paragraph("Values with 95% bootstrap confidence intervals. The screening "
                           "threshold on P(Malignant) was chosen on val for 95% Malignant sensitivity.",
                           SMALL))
    story.append(Spacer(1, 6))
    story.append(fig("fig4_test_confusion.png", 17.2))
    story.append(KeepTogether([
        Paragraph("Conclusions", H2),
        Paragraph(
            "• The baseline clearly beats all dummies: test macro-F1 0.697 [0.589, 0.787] and macro AUC "
            "0.879 vs 0.33 and 0.50.<br/>"
            "• Colour and texture of both views are the most useful cues; shape and metadata add a little; "
            "HOG adds noise; the dermoscopic filter bank is redundant once texture is present.<br/>"
            "• Gradient boosting and SVM-RBF are the strongest classical models; their soft-voting "
            "ensemble generalises best on val.<br/>"
            "• As a triage tool at 97.6% Malignant sensitivity it still spares 44% of benign lesions, "
            "but it is not confident enough to rule out malignancy under the 50:1 cost matrix.<br/>"
            "• Uncertainty is large (13 Indeterminate test lesions; a tiny feature perturbation moved test "
            "macro-F1 from 0.697 to 0.722), so differences below about 0.02 are noise. The CV estimate "
            "(about 0.61) is the most reliable single number. This is the reference a CNN must beat in "
            "Milestone 2.",
            BODY)]))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Homework 5 — Classical ML Baseline (MILK10k)",
                            author="Daniela Gonzalez")
    doc.build(story)
    print("Wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
