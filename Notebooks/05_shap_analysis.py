"""
shap_analysis.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
Step 5: SHAP interpretability analysis using TreeExplainer on Random Forest.
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR   = os.path.join(OUTPUT_DIR, "data")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR= os.path.join(OUTPUT_DIR, "results")
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("SHAP INTERPRETABILITY ANALYSIS")
print("=" * 60)

# ── Load model and test data ───────────────────────────────────
print("Loading Random Forest model...")
rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

print("Loading test set...")
test = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
X_test = test.drop("label", axis=1)
y_test = test["label"]
print(f"  Test set: {X_test.shape[0]} rows x {X_test.shape[1]} features")

# Sample 500 rows for SHAP (computational efficiency)
idx = np.random.choice(len(X_test), size=500, replace=False)
X_sample = X_test.iloc[idx].reset_index(drop=True)
print(f"  Using {len(X_sample)} randomly sampled test rows for SHAP")

# ── Compute SHAP values ────────────────────────────────────────
print("\nComputing SHAP values (this may take 1-2 minutes)...")
explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# For binary classification, use class 1 (arrhythmia)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

print(f"  SHAP values shape: {sv.shape}")

# ── Plot: Beeswarm summary ─────────────────────────────────────
print("\nGenerating beeswarm summary plot...")
plt.figure(figsize=(10, 8))
plt.title("SHAP Feature Impact -- Random Forest\n(each dot = one patient; red = high feature value, blue = low)",
          fontsize=11, fontweight="bold")
shap.summary_plot(sv, X_sample, show=False, max_display=15)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_summary_beeswarm.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'shap_summary_beeswarm.png')}")

# ── Plot: Bar chart top 15 ─────────────────────────────────────
print("Generating top-15 feature importance bar chart...")
mean_shap = np.abs(sv).mean(axis=0)
feat_imp  = pd.Series(mean_shap, index=X_sample.columns).sort_values(ascending=False)
top15     = feat_imp.head(15)

print("\nTop 10 features:")
for feat, val in top15.head(10).items():
    print(f"  {feat:<30} {val:.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 15))[::-1]
bars = ax.barh(top15.index[::-1], top15.values[::-1], color=colors)
for bar, val in zip(bars, top15.values[::-1]):
    ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
ax.set_title("Top 15 Most Important ECG Features\n(SHAP — Random Forest)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_bar_top15.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.join(PLOTS_DIR, 'shap_bar_top15.png')}")

# ── Save full feature importance CSV ──────────────────────────
feat_imp.reset_index().rename(columns={"index":"feature", 0:"mean_abs_shap"}).to_csv(
    os.path.join(RESULTS_DIR, "shap_feature_importance.csv"), index=False)
print(f"\nFull feature importance saved: {os.path.join(RESULTS_DIR, 'shap_feature_importance.csv')}")

print("\nAll done! Check outputs/plots/ for poster-ready figures.")
print("\n[DONE] Step 5 complete.")

