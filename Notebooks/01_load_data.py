"""
load_data.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
Step 1: Load PTB-XL Plus dataset, explore class distribution, create stratified splits.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── Configuration ─────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1 — DATA LOADING & EXPLORATION")
print("=" * 60)

# ── Load metadata ─────────────────────────────────────────────
meta_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
if not os.path.exists(meta_path):
    # Try nested path
    meta_path = os.path.join(DATA_DIR,
        "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1",
        "ptbxl_database.csv")

meta = pd.read_csv(meta_path, index_col="ecg_id")
print(f"Metadata loaded: {meta.shape[0]} records")

# ── Load features ─────────────────────────────────────────────
feat_path = os.path.join(DATA_DIR, "features", "12sl_features.csv")
if not os.path.exists(feat_path):
    feat_path = os.path.join(DATA_DIR,
        "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1",
        "features", "12sl_features.csv")

features = pd.read_csv(feat_path)
assert len(features) == len(meta), f"Row mismatch: meta={len(meta)}, features={len(features)}"
features.index = meta.index
features.index.name = "ecg_id"
print(f"Features loaded: {features.shape}")

# ── Merge metadata with features ──────────────────────────────
df = meta.join(features, how="inner")
print(f"Merged dataset: {df.shape}")



# ── Build binary arrhythmia label (rhythm-only, clinically correct) ──
# The old rule labeled ANY non-normal code as "arrhythmia", which wrongly
# caught heart attacks (MI), ST/T changes, hypertrophy, etc.
# We now use PTB-XL's own scp_statements.csv to identify RHYTHM codes,
# then treat sinus-family rhythms (SR, STACH, SBRAD, SARRH) as normal.
import ast

# Load the code categories (which codes are rhythm vs diagnostic vs form)
scp_stmt_path = os.path.join(DATA_DIR, "scp_statements.csv")
if not os.path.exists(scp_stmt_path):
    scp_stmt_path = os.path.join(DATA_DIR,
        "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1",
        "scp_statements.csv")

scp_stmt = pd.read_csv(scp_stmt_path, index_col=0)
RHYTHM_CODES = set(scp_stmt.index[scp_stmt["rhythm"] == 1.0])
print(f"Rhythm codes from scp_statements.csv ({len(RHYTHM_CODES)}): "
      f"{sorted(RHYTHM_CODES)}")

# Sinus-family = normal rhythm (NOT arrhythmia).
# Everything else in RHYTHM_CODES (AFIB, AFLT, PACE, PVC-related, etc.) = arrhythmia.
NORMAL_RHYTHM = {"SR", "STACH", "SBRAD", "SARRH"}

def is_arrhythmia(scp_str):
    """1 = record has a non-sinus rhythm code; 0 = otherwise."""
    try:
        scp_dict = ast.literal_eval(scp_str)
        codes = set(scp_dict.keys())
        arrhythmia_codes = (codes & RHYTHM_CODES) - NORMAL_RHYTHM
        return 1 if arrhythmia_codes else 0
    except Exception:
        return 0

df["label"] = df["scp_codes"].apply(is_arrhythmia)

# ── Class distribution ────────────────────────────────────────
counts = df["label"].value_counts()
n_normal = counts[0]
n_arrhythmia = counts[1]
total = len(df)

print(f"\nClass distribution:")
print(f"  Non-arrhythmia: {n_normal} ({n_normal/total*100:.1f}%)")
print(f"  Arrhythmia:     {n_arrhythmia} ({n_arrhythmia/total*100:.1f}%)")
print(f"  Total:          {total}")

# ── Arrhythmia subtypes ────────────────────────────────────────
ARRHYTHMIA_MAP = {
    "AFIB":"Atrial fibrillation","AFLT":"Atrial flutter",
    "SVTAC":"Supraventricular tachy.","PSVT":"Paroxysmal SVT",
    "STACH":"Sinus tachycardia","SBRAD":"Sinus bradycardia",
    "SARRH":"Sinus arrhythmia","1AVB":"1st-degree AV block",
    "2AVB":"2AVB","3AVB":"3AVB","PACE":"Pacemaker rhythm",
    "PVC":"Premature vent. contractions","BIGU":"BIGU",
    "TRIGU":"TRIGU","WPW":"Wolff-Parkinson-White","SVARR":"SVARR",
}

subtype_counts = Counter()
for scp_str in df[df["label"]==1]["scp_codes"]:
    try:
        scp_dict = ast.literal_eval(scp_str)
        for code in scp_dict:
            if code in RHYTHM_CODES and code not in NORMAL_RHYTHM:
                subtype_counts[code] += 1
    except Exception:
        pass

top_subtypes = pd.Series({
    ARRHYTHMIA_MAP.get(k, k): v
    for k, v in subtype_counts.most_common(16)
})

# ── Plot 1: Class distribution ─────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PTB-XL+ Dataset — Arrhythmia vs Non-arrhythmia", fontsize=14, fontweight="bold")

bars = ax1.bar(["Non-arrhythmia","Arrhythmia"], [n_normal, n_arrhythmia],
               color=["#4472C4","#ED7D31"], edgecolor="white", width=0.5)
ax1.set_title("Class Distribution", fontweight="bold")
ax1.set_ylabel("Number of ECG Recordings")
for bar, val, pct in zip(bars, [n_normal, n_arrhythmia],
                          [n_normal/total*100, n_arrhythmia/total*100]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f"{val:,}\n({pct:.1f}%)", ha="center", fontweight="bold", fontsize=11)
ax1.set_ylim(0, n_normal * 1.15)

ax2.pie(
    [n_normal, n_arrhythmia],
    labels=[f"Non-arrhythmia\n{n_normal:,} ({n_normal/total*100:.1f}%)",
            f"Arrhythmia\n{n_arrhythmia:,} ({n_arrhythmia/total*100:.1f}%)"],
    colors=["#4472C4","#ED7D31"], startangle=90,
    wedgeprops={"edgecolor":"white","linewidth":2}
)
ax2.set_title("Class Proportions", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: class_distribution.png")

# ── Plot 2: Arrhythmia subtypes ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top_subtypes)))
bars = ax.barh(top_subtypes.index[::-1], top_subtypes.values[::-1], color=colors[::-1])
for bar, val in zip(bars, top_subtypes.values[::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10)
ax.set_xlabel("Number of ECG Recordings", fontsize=12)
ax.set_title("Arrhythmia Subtypes in PTB-XL+", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "arrhythmia_subtypes.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: arrhythmia_subtypes.png")

# ── Stratified splits ─────────────────────────────────────────
feature_cols = [c for c in features.columns if c in df.columns]
df_clean = df[feature_cols + ["label", "strat_fold"]].copy()

train_df = df_clean[df_clean["strat_fold"] <= 8].drop("strat_fold", axis=1)
val_df   = df_clean[df_clean["strat_fold"] == 9].drop("strat_fold", axis=1)
test_df  = df_clean[df_clean["strat_fold"] == 10].drop("strat_fold", axis=1)

print(f"\nSplits:")
print(f"  Train: {len(train_df)} ({train_df['label'].mean()*100:.1f}% arrhythmia)")
print(f"  Val:   {len(val_df)} ({val_df['label'].mean()*100:.1f}% arrhythmia)")
print(f"  Test:  {len(test_df)} ({test_df['label'].mean()*100:.1f}% arrhythmia)")

# Save
os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)
train_df.to_csv(os.path.join(OUTPUT_DIR, "data", "train_raw.csv"))
val_df.to_csv(os.path.join(OUTPUT_DIR, "data", "val_raw.csv"))
test_df.to_csv(os.path.join(OUTPUT_DIR, "data", "test_raw.csv"))
print("\nData splits saved to outputs/data/")
print("\n✅ Step 1 complete.")


