"""
generate_milestone1_report.py
B9: generate reports/milestone1_report.pdf (max 2 pages).
"""
from pathlib import Path
import json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                PageBreak, Image, Table, TableStyle)
from reportlab.lib.enums import TA_JUSTIFY

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "milestone1_report.pdf"
FIG = ROOT / "reports" / "figures_milestone1"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15,
                    textColor=HexColor("#1f3a5f"), spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11,
                    textColor=HexColor("#1f3a5f"), spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=9,
                      leading=12, alignment=TA_JUSTIFY, spaceAfter=6)

story = []

story.append(Paragraph("MILK10k — Course Project Milestone 1: Data Pipeline", H1))
story.append(Paragraph(
    "<b>Student:</b> Daniela Gonzalez &nbsp;&nbsp; "
    "<b>Repository:</b> https://github.com/Dgbaquero123/milk10k-eda",
    BODY))
story.append(Spacer(1, 0.2 * cm))

story.append(Paragraph("1. Label strategy", H2))
story.append(Paragraph(
    "The primary target is <b>diagnosis_1</b> with three classes: "
    "Benign (1,483 lesions), Indeterminate (123), Malignant (3,634). "
    "Indeterminate is kept as its own class because it is a real "
    "decision boundary in the clinical workflow (refer vs reassure) and "
    "123 lesions is small but trainable with class weights. The 11-class "
    "stretch goal has a long tail: VASC (47), BEN_OTH (44) and MAL_OTH "
    "(9) are merged into a single <font face='Courier'>other</font> bucket "
    "because a class with fewer than ~50 lesions cannot be reliably "
    "measured on a 70/15/15 split (see A3.2). The final mapping is saved "
    "to <font face='Courier'>configs/label_map.json</font>.", BODY))

story.append(Paragraph("2. Split design and verification", H2))
story.append(Paragraph(
    "Splits are at the <b>lesion</b> level, grouped by <font face='Courier'>"
    "lesion_id</font> and stratified on the 11-class label. A greedy "
    "assignment with a seeded shuffle produces exactly 70 / 15 / 15. "
    "Verification across the 10 seeds and across the three CSV files:", BODY))

verif_data = [
    ["check", "result"],
    ["train / val / test sizes", "3,668 / 786 / 786 lesions (70 / 15 / 15 %)"],
    ["zero lesion overlap between splits", "confirmed (assert + pytest)"],
    ["each lesion has exactly 2 images", "confirmed for all 5,240"],
    ["max class-proportion deviation", "0.0212 (BCC)"],
    ["reproducibility (same seed)", "confirmed (pytest)"],
    ["StratifiedGroupKFold vs alternatives", "0 leaked lesions; alternatives leaked 4,235"],
]
t = Table(verif_data, colWidths=[7*cm, 9*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), HexColor("#1f3a5f")),
    ("TEXTCOLOR", (0,0), (-1,0), HexColor("#ffffff")),
    ("GRID", (0,0), (-1,-1), 0.3, HexColor("#888888")),
    ("FONTSIZE", (0,0), (-1,-1), 8),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(t)
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("3. Imbalance ratio on TRAIN", H2))
story.append(Paragraph(
    "On the train split the ratio between the most and least frequent "
    "class is <b>28.4:1</b> (Malignant 2,556 / Indeterminate 90). Class "
    "weights computed as <font face='Courier'>n / (n_classes * count)</font> "
    "give <font face='Courier'>[1.195, 13.585, 0.479]</font> for "
    "Benign / Indeterminate / Malignant. These weights are saved to "
    "<font face='Courier'>configs/class_weights.json</font>. A "
    "WeightedRandomSampler with the same weights is available as an "
    "alternative and is implemented in <font face='Courier'>dataset.py</font>.",
    BODY))

story.append(Paragraph("4. Quality issues found and how they were handled", H2))
story.append(Paragraph(
    "<b>Integrity.</b> All 10,480 images resolve to a file and pass "
    "<font face='Courier'>Image.verify()</font>. All images are 600×450. "
    "<b>Missing values.</b> age_approx (0.38%) imputed with the train "
    "median; anatom_site_general (37%) kept as 'unknown'; "
    "anatom_site_special (98%) dropped; melanocytic (77%) dropped. "
    "<b>Label leakage.</b> <font face='Courier'>diagnosis_2/3/4</font> "
    "and <font face='Courier'>diagnosis_confirm_type</font> are excluded "
    "from model inputs because they encode the target. "
    "<b>Shortcuts.</b> image_manipulation is not uniform across classes "
    "(altered: 39.7% Benign vs instrument-only: 27.9%), so it is not used "
    "as a feature without ablation. File size was checked as a shortcut "
    "and has AUC ≈ 0.5, so it is not exploitable.", BODY))

story.append(PageBreak())

story.append(Paragraph("5. Preprocessing choices", H2))
story.append(Paragraph(
    "<b>Resolution: 224 × 224.</b> All images are 600 × 450, so no image "
    "is lost by resizing. 224 matches ImageNet-pretrained CNN inputs. "
    "<b>Views: both</b> (dermoscopic + clinical), one tensor each, "
    "matching the lesion-level unit. <b>Normalization: per-channel "
    "train-only mean/std</b> = "
    "<font face='Courier'>[0.679, 0.527, 0.475]</font> / "
    "<font face='Courier'>[0.127, 0.134, 0.156]</font>, saved to "
    "<font face='Courier'>configs/norm_stats.json</font>. Validation and "
    "test use the same statistics, computed only on train. "
    "<b>Augmentations</b> (train only): flips, rotation ±15°, "
    "ColorJitter(brightness 0.1, contrast 0.1, saturation 0.05, "
    "hue 0.02), RandomResizedCrop(scale 0.85-1.0). Parameters were chosen "
    "from the A3.5 colour-safety audit: the jitter stays well inside the "
    "Benign-vs-Malignant hue gap (42°). <font face='Courier'>eval_transform"
    "</font> is deterministic (verified).", BODY))

story.append(Paragraph("6. What could still go wrong", H2))
story.append(Paragraph(
    "Three residual risks matter for Milestone 2. <b>(i) Group leakage "
    "at the patient level.</b> MILK10k uses <font face='Courier'>"
    "lesion_id</font> as the grouping unit, but two different lesions "
    "can belong to the same patient. If MILK10k contains multiple "
    "lesions per patient, splitting by lesion_id is not enough to "
    "prevent patient-level leakage. This should be checked in a future "
    "milestone by looking for repeated demographic signatures. "
    "<b>(ii) Sampling bias.</b> 95.7% of the images are histopathology-"
    "confirmed, which means the dataset is biopsy-enriched and does not "
    "reflect real-world prevalence. A model that achieves high accuracy "
    "here is not a prevalence estimator. <b>(iii) What a clinician "
    "should know.</b> The model would act as a second reader on lesions "
    "that a clinician has already decided to biopsy. It does not tell "
    "the clinician how worried to be about a lesion that has not yet "
    "been flagged, and its predictions should always be paired with the "
    "cost matrix from A3.4 rather than with a single accuracy number.",
    BODY))

story.append(Spacer(1, 0.3*cm))
if (FIG / "b2_diagnosis_1_distribution.png").exists():
    story.append(Image(str(FIG / "b2_diagnosis_1_distribution.png"),
                       width=6*cm, height=3.5*cm))
if (FIG / "b2_gallery_3x4.png").exists():
    story.append(Image(str(FIG / "b2_gallery_3x4.png"),
                       width=10*cm, height=6*cm))

doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=1.8*cm, rightMargin=1.8*cm,
                        topMargin=1.5*cm, bottomMargin=1.5*cm,
                        title="MILK10k Milestone 1 Report",
                        author="Daniela Gonzalez")
doc.build(story)
print(f"PDF generated: {OUT}")
print(f"Size: {OUT.stat().st_size / 1024:.1f} KB")
