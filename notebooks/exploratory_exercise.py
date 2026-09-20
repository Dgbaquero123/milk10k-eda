from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_DIR = Path("data/raw/milk10k")
SUPP_DIR = DATA_DIR / "supplements"
FIG_DIR  = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
meta = pd.read_csv(DATA_DIR / "metadata.csv")
gt   = pd.read_csv(SUPP_DIR / "training_gt.csv")

# One row per lesion (each lesion has 2 images)
les = meta.drop_duplicates("lesion_id").copy()

# ============================================================
# 1) Number of samples per diagnosis_1 — BAR PLOT
# ============================================================
dx1 = les["diagnosis_1"].value_counts().reindex(
    ["Benign", "Malignant", "Indeterminate"]).dropna()

fig, ax = plt.subplots(figsize=(6, 4))
dx1.plot(kind="bar", ax=ax, color=["#2E8B57", "#B22222", "#808080"])
ax.set_ylabel("Number of samples")
ax.set_title("MILK10k — Samples per diagnosis_1 (3 classes)")
for i, v in enumerate(dx1.values):
    ax.text(i, v + 30, str(v), ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "ex1_diagnosis1_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 2) Number of samples per subclass (11 diagnoses) — BAR PLOT
# ============================================================
gt_long = (gt.set_index("lesion_id").stack()
             .loc[lambda s: s == 1]
             .reset_index()[["lesion_id", "level_1"]]
             .rename(columns={"level_1": "class"}))
class_counts = gt_long["class"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
class_counts.plot(kind="bar", ax=ax, color="#F08200")
ax.set_ylabel("Number of samples")
ax.set_title("MILK10k — Samples per subclass (11 diagnoses)")
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 30, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "ex2_subclass_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 3) Age range per class — TABLE
# ============================================================
age_table = (les.groupby("diagnosis_1")["age_approx"]
                .agg(["min", "max", "mean", "median", "count"])
                .round(2))
age_table.columns = ["min_age", "max_age", "mean_age", "median_age", "n_samples"]
print("\n=== Age range per class (diagnosis_1) ===")
print(age_table)

age_table.to_csv(FIG_DIR / "ex3_age_table.csv")

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=les, x="diagnosis_1", y="age_approx", ax=ax)
ax.set_title("Age distribution per diagnosis_1")
plt.tight_layout()
plt.savefig(FIG_DIR / "ex3_age_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 4) Sex distribution per class — TABLE
# ============================================================
sex_table = pd.crosstab(les["diagnosis_1"], les["sex"], margins=True, margins_name="Total")
print("\n=== Sex distribution per class (diagnosis_1) ===")
print(sex_table)

sex_table.to_csv(FIG_DIR / "ex4_sex_table.csv")

fig, ax = plt.subplots(figsize=(7, 4))
sex_table.drop(index="Total").drop(columns="Total").plot(
    kind="bar", stacked=True, ax=ax, color=["#E48FB6", "#6FA8DC"])
ax.set_ylabel("Number of samples")
ax.set_title("Sex distribution per diagnosis_1")
plt.tight_layout()
plt.savefig(FIG_DIR / "ex4_sex_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nDone. Figures and tables saved to reports/figures/")
