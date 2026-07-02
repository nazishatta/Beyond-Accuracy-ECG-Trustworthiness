"""
evaluate.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
Step 4: Trustworthiness evaluation — sensitivity, PR curves, calibration, false-negative analysis.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              precision_recall_curve, average_precision_score,
                              brier_score_loss)
from sklearn.calibration import calibration_curve

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
print("STEP 4 — TRUSTWORTHINESS EVALUATION")
print("=" * 60)

# ── Load test data ─────────────────────────────────────────────
test = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
X_test = test.drop("label", axis=1)
y_test = test["label"]
print(f"Test set: {X_test.shape} | Arrhythmia: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ── Load models ────────────────────────────────────────────────
model_names = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}
models = {v: joblib.load(os.path.join(MODELS_DIR, f"{k}.pkl"))
          for k, v in model_names.items()}

# ── Evaluate all models ────────────────────────────────────────
results = []
preds   = {}
probs   = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    preds[name] = y_pred
    probs[name] = y_prob

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Model":       name,
        "Accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "Precision":   round(precision_score(y_test, y_pred), 4),
        "Recall":      round(recall_score(y_test, y_pred), 4),
        "Specificity": round(tn / (tn + fp), 4),
        "F1":          round(f1_score(y_test, y_pred), 4),
        "AUC_ROC":     round(roc_auc_score(y_test, y_prob), 4),
        "Avg_Precision": round(average_precision_score(y_test, y_prob), 4),
        "Brier_Score": round(brier_score_loss(y_test, y_prob), 4),
        "FN": int(fn),
        "FP": int(fp),
        "TP": int(tp),
        "TN": int(tn),
    })
    print(f"\n{name}:")
    print(f"  Accuracy={results[-1]['Accuracy']:.4f} | Recall={results[-1]['Recall']:.4f} | "
          f"Specificity={results[-1]['Specificity']:.4f} | AUC={results[-1]['AUC_ROC']:.4f} | "
          f"Brier={results[-1]['Brier_Score']:.4f} | FN={fn} | FP={fp}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
print(f"\nResults saved to outputs/results/metrics_comparison.csv")

COLORS = {"Logistic Regression": "#4472C4", "Random Forest": "#ED7D31", "XGBoost": "#70AD47"}

# ── Plot: Precision-Recall curves ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
for name, model in models.items():
    p, r, _ = precision_recall_curve(y_test, probs[name])
    ap = average_precision_score(y_test, probs[name])
    ax.plot(r, p, label=f"{name} (AP={ap:.3f})", color=COLORS[name], lw=2)
baseline = y_test.mean()
ax.axhline(baseline, color="gray", linestyle="--", label=f"Random baseline ({baseline:.2f})")
ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — Test Set", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pr_curve_all_models.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: pr_curve_all_models.png")

# ── Plot: Calibration curves ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1], "k--", label="Perfect calibration")
for name, model in models.items():
    frac_pos, mean_pred = calibration_curve(y_test, probs[name], n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", label=name, color=COLORS[name], lw=2)
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives (Actual)", fontsize=12)
ax.set_title("Calibration Curves — Test Set\n(closer to diagonal = better calibrated)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "calibration_curve.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: calibration_curve.png")

# ── Plot: False negative analysis (Random Forest) ─────────────
rf_name = "Random Forest"
rf_prob = probs[rf_name]
rf_pred = preds[rf_name]

mask_arrhythmia = y_test == 1
detected = (rf_pred == 1) & mask_arrhythmia
missed   = (rf_pred == 0) & mask_arrhythmia

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("False-Negative Analysis — Random Forest (Test Set)", fontsize=14, fontweight="bold")

ax1.hist(rf_prob[detected], bins=20, alpha=0.7, color="#70AD47", label=f"Correctly detected (n={detected.sum()})")
ax1.hist(rf_prob[missed],   bins=20, alpha=0.7, color="#ED7D31", label=f"Missed / false negative (n={missed.sum()})")
ax1.axvline(0.5, color="black", linestyle="--", label="Decision threshold (0.5)")
ax1.set_xlabel("Predicted Probability of Arrhythmia", fontsize=11)
ax1.set_ylabel("Count", fontsize=11)
ax1.set_title("Predicted Probability Distribution\n(Actual Arrhythmia Cases Only)", fontweight="bold")
ax1.legend(fontsize=9)

bins = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
labels_bins = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
det_counts  = [((rf_prob >= bins[i]) & (rf_prob < bins[i+1]) & detected).sum() for i in range(len(bins)-1)]
miss_counts = [((rf_prob >= bins[i]) & (rf_prob < bins[i+1]) & missed).sum()   for i in range(len(bins)-1)]
x = np.arange(len(labels_bins)); w = 0.35
ax2.bar(x-w/2, det_counts,  w, label="Detected", color="#70AD47", alpha=0.8)
ax2.bar(x+w/2, miss_counts, w, label="Missed",   color="#ED7D31", alpha=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(labels_bins, rotation=45, ha="right", fontsize=9)
ax2.set_xlabel("Predicted Probability Bucket", fontsize=11)
ax2.set_ylabel("Number of Arrhythmia Cases", fontsize=11)
ax2.set_title("Detected vs Missed Arrhythmias\nby Predicted Probability", fontweight="bold")
ax2.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "false_negative_analysis.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: false_negative_analysis.png")

print("\n[DONE] Step 4 complete.")

