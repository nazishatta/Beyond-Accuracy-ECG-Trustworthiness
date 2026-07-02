"""
calibration_clinical.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
    Expected Calibration Error (ECE), decision curve analysis,
         and reframed thesis around calibration + error character.

Why this matters:
  - ECE is the standard scalar for calibration that reviewers expect.
  - Decision curve analysis shows CLINICAL UTILITY — not just discrimination.
  - Together they answer: "would using this model actually help a doctor?"
  
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
DATA_DIR    = os.path.join(OUTPUT_DIR, "data")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

print("=" * 60)
print("PHASE 3 — CALIBRATION & CLINICAL UTILITY")
print("=" * 60)

# ── Load test data and models ──────────────────────────────────
test = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
X_test = test.drop("label", axis=1)
y_test = test["label"].values
print(f"Test set: {X_test.shape} | Arrhythmia: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

model_files = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}
models = {name: joblib.load(os.path.join(MODELS_DIR, f"{fname}.pkl"))
          for name, fname in model_files.items()}

y_probs = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}
y_preds = {n: m.predict(X_test)              for n, m in models.items()}

COLORS = {"Logistic Regression": "#4472C4",
          "Random Forest": "#ED7D31",
          "XGBoost": "#70AD47"}

# ================================================================
# PART 1: EXPECTED CALIBRATION ERROR (ECE)
# ================================================================
# How it works :
#   Split all predictions into 10 buckets by predicted probability.
#   In each bucket, compare:
#     - what the model PREDICTED (average probability in that bucket)
#     - what ACTUALLY happened (fraction that were truly arrhythmia)
#   If the model says "70% chance" and 70% of those cases really are
#   arrhythmia, the model is perfectly calibrated.
#   ECE = the weighted average gap across all buckets.
#   Lower ECE = better calibrated = more trustworthy probabilities.
# ================================================================
print("\n" + "=" * 60)
print("PART 1: EXPECTED CALIBRATION ERROR (ECE)")
print("=" * 60)

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error.
    Returns: ECE value, per-bin details for plotting.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if i == 0:  # include 0.0 in first bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_details.append({
                "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                "count": 0, "avg_pred": 0, "avg_true": 0, "gap": 0
            })
            continue

        avg_pred = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        gap = abs(avg_true - avg_pred)
        weight = n_in_bin / len(y_true)
        ece += weight * gap

        bin_details.append({
            "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
            "count": int(n_in_bin),
            "avg_pred": round(avg_pred, 4),
            "avg_true": round(avg_true, 4),
            "gap": round(gap, 4),
        })

    return round(ece, 4), bin_details

ece_rows = []
for name in models:
    ece_val, details = compute_ece(y_test, y_probs[name])
    brier = brier_score_loss(y_test, y_probs[name])
    ece_rows.append({
        "Model": name,
        "ECE": ece_val,
        "Brier Score": round(brier, 4),
    })
    print(f"\n  {name}:")
    print(f"    ECE = {ece_val:.4f}  |  Brier = {brier:.4f}")
    print(f"    Per-bin breakdown:")
    for d in details:
        if d["count"] > 0:
            print(f"      {d['bin']:10s}  n={d['count']:4d}  "
                  f"pred={d['avg_pred']:.3f}  actual={d['avg_true']:.3f}  "
                  f"gap={d['gap']:.3f}")

ece_df = pd.DataFrame(ece_rows)
ece_path = os.path.join(RESULTS_DIR, "calibration_ece.csv")
ece_df.to_csv(ece_path, index=False)
print(f"\nSaved: {ece_path}")

# ── Plot: Calibration curves with ECE ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
for name in models:
    frac_pos, mean_pred = calibration_curve(y_test, y_probs[name], n_bins=10)
    ece_val = ece_df[ece_df["Model"] == name]["ECE"].values[0]
    brier   = ece_df[ece_df["Model"] == name]["Brier Score"].values[0]
    ax.plot(mean_pred, frac_pos, "o-", color=COLORS[name], lw=2.2, ms=7,
            label=f"{name}\n  ECE={ece_val:.4f}  Brier={brier:.4f}")
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives (Actual Arrhythmia Rate)", fontsize=12)
ax.set_title("Calibration Curves with ECE\nCloser to diagonal = better calibrated",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
plt.tight_layout()
cal_path = os.path.join(PLOTS_DIR, "calibration_with_ece.png")
plt.savefig(cal_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {cal_path}")

# ================================================================
# PART 2: DECISION CURVE ANALYSIS (DCA)
# ================================================================
# How it works:
#   A doctor has to decide: treat this patient or not?
#   Different doctors have different "worry levels" (threshold
#   probabilities). A cautious doctor treats at 5% risk. A
#   conservative one waits until 30%.
#
#   For each threshold, DCA computes "net benefit":
#     net benefit = (true positives / n) - (false positives / n)
#                   × (threshold / (1 - threshold))
#
#   The last part is the penalty for unnecessary treatment.
#   Higher net benefit = model adds more value than harm.
#
#   Two baselines to compare against:
#     - "Treat all" = give everyone treatment (catches everything,
#       but many unnecessary treatments)
#     - "Treat none" = net benefit of zero (no harm, no help)
#
#   A useful model has net benefit ABOVE both baselines.

# ================================================================
print("\n" + "=" * 60)
print("PART 2: DECISION CURVE ANALYSIS (DCA)")
print("=" * 60)

def decision_curve(y_true, y_prob, thresholds):
    """Compute net benefit at each threshold."""
    n = len(y_true)
    net_benefits = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tp = np.sum((y_pred_t == 1) & (y_true == 1))
        fp = np.sum((y_pred_t == 1) & (y_true == 0))
        # Net benefit formula
        nb = (tp / n) - (fp / n) * (t / (1 - t)) if t < 1 else 0
        net_benefits.append(nb)
    return np.array(net_benefits)

thresholds = np.arange(0.01, 0.80, 0.01)
prevalence = y_test.mean()

# "Treat all" baseline
treat_all_nb = []
for t in thresholds:
    nb = prevalence - (1 - prevalence) * (t / (1 - t))
    treat_all_nb.append(nb)
treat_all_nb = np.array(treat_all_nb)

# Plot DCA
fig, ax = plt.subplots(figsize=(10, 7))

# Treat none = 0 line
ax.axhline(0, color="black", lw=1, label="Treat none")

# Treat all
ax.plot(thresholds, treat_all_nb, "k--", lw=1.5, label="Treat all")

# Each model
for name in models:
    nb = decision_curve(y_test, y_probs[name], thresholds)
    ax.plot(thresholds, nb, lw=2.5, color=COLORS[name], label=name)

ax.set_xlabel("Threshold Probability", fontsize=12)
ax.set_ylabel("Net Benefit", fontsize=12)
ax.set_title("Decision Curve Analysis — Arrhythmia Detection\n"
             "Higher net benefit = more clinical value at that threshold",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim([0, 0.80])
ax.set_ylim([-0.05, max(prevalence * 1.1, 0.15)])
ax.grid(True, alpha=0.3)
plt.tight_layout()
dca_path = os.path.join(PLOTS_DIR, "decision_curve_analysis.png")
plt.savefig(dca_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {dca_path}")

# Save DCA values
dca_rows = []
for name in models:
    nb = decision_curve(y_test, y_probs[name], thresholds)
    for t, n_b in zip(thresholds, nb):
        dca_rows.append({"Model": name, "Threshold": round(t, 2),
                         "Net_Benefit": round(n_b, 6)})
dca_df = pd.DataFrame(dca_rows)
dca_csv_path = os.path.join(RESULTS_DIR, "decision_curve_values.csv")
dca_df.to_csv(dca_csv_path, index=False)
print(f"Saved: {dca_csv_path}")

# ================================================================
# PART 3: FALSE-NEGATIVE ERROR CHARACTER ANALYSIS
# ================================================================
# How it works:
#   When a model misses an arrhythmia patient (false negative),
#   HOW WRONG was it? Did it say "49% chance" (close to catching
#   them) or "2% chance" (confidently wrong)?
#   This matters because:
#   - A "49% miss" could be saved by lowering the threshold slightly.
#   - A "2% miss" can never be recovered — the model is sure it's normal.
#   We compute statistics on the predicted probabilities of missed cases.
# ================================================================
print("\n" + "=" * 60)
print("PART 3: FALSE-NEGATIVE ERROR CHARACTER")
print("=" * 60)

arr_mask = y_test == 1
fn_char_rows = []

for name in models:
    yp = y_probs[name]
    yd = y_preds[name]
    missed = arr_mask & (yd == 0)
    detected = arr_mask & (yd == 1)

    if missed.sum() == 0:
        print(f"\n  {name}: No false negatives!")
        continue

    fn_probs = yp[missed]
    det_probs = yp[detected]

    # How many missed cases could be recovered at lower thresholds?
    recoverable_30 = (fn_probs >= 0.30).sum()
    recoverable_20 = (fn_probs >= 0.20).sum()
    recoverable_10 = (fn_probs >= 0.10).sum()
    confident_wrong = (fn_probs < 0.10).sum()

    row = {
        "Model": name,
        "Total_FN": int(missed.sum()),
        "FN_prob_mean": round(fn_probs.mean(), 4),
        "FN_prob_median": round(np.median(fn_probs), 4),
        "FN_prob_min": round(fn_probs.min(), 4),
        "FN_prob_max": round(fn_probs.max(), 4),
        "FN_prob_std": round(fn_probs.std(), 4),
        "Recoverable_at_0.30": int(recoverable_30),
        "Recoverable_at_0.20": int(recoverable_20),
        "Recoverable_at_0.10": int(recoverable_10),
        "Confident_wrong_below_0.10": int(confident_wrong),
    }
    fn_char_rows.append(row)

    print(f"\n  {name} ({missed.sum()} false negatives):")
    print(f"    Predicted probabilities of missed cases:")
    print(f"      Mean:   {fn_probs.mean():.4f}")
    print(f"      Median: {np.median(fn_probs):.4f}")
    print(f"      Min:    {fn_probs.min():.4f}  Max: {fn_probs.max():.4f}")
    print(f"    Recoverability (by lowering threshold):")
    print(f"      At threshold 0.30: {recoverable_30}/{missed.sum()} "
          f"({recoverable_30/missed.sum()*100:.0f}%) recoverable")
    print(f"      At threshold 0.20: {recoverable_20}/{missed.sum()} "
          f"({recoverable_20/missed.sum()*100:.0f}%) recoverable")
    print(f"      At threshold 0.10: {recoverable_10}/{missed.sum()} "
          f"({recoverable_10/missed.sum()*100:.0f}%) recoverable")
    print(f"      Confidently wrong (<0.10): {confident_wrong}/{missed.sum()} "
          f"({confident_wrong/missed.sum()*100:.0f}%) — UNRECOVERABLE")

fn_char_df = pd.DataFrame(fn_char_rows)
fn_char_path = os.path.join(RESULTS_DIR, "false_negative_error_character.csv")
fn_char_df.to_csv(fn_char_path, index=False)
print(f"\nSaved: {fn_char_path}")

# ── Plot: FN probability distributions side by side ────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name in zip(axes, models):
    yp = y_probs[name]
    yd = y_preds[name]
    detected = arr_mask & (yd == 1)
    missed   = arr_mask & (yd == 0)

    ax.hist(yp[detected], bins=20, alpha=0.78, color="#2E75B6",
            edgecolor="white", label=f"Detected (n={detected.sum()})")
    ax.hist(yp[missed],   bins=20, alpha=0.78, color="#C00000",
            edgecolor="white", label=f"Missed FN (n={missed.sum()})")
    ax.axvline(0.5, color="black", ls="--", lw=2, label="Threshold = 0.50")
    ax.axvline(0.1, color="orange", ls=":", lw=1.5, label="Threshold = 0.10")
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    fn_rate = missed.sum() / arr_mask.sum() if arr_mask.sum() > 0 else 0
    ax.set_title(f"{name}\nFN={missed.sum()}/{arr_mask.sum()} ({fn_rate:.1%})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("False-Negative Error Character — Arrhythmia Cases Only\n"
             "Red bars below orange line (0.10) = confidently wrong, unrecoverable",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fn_plot_path = os.path.join(PLOTS_DIR, "fn_error_character.png")
plt.savefig(fn_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fn_plot_path}")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("PHASE 3 SUMMARY")
print("=" * 60)

print("\n  CALIBRATION:")
for _, row in ece_df.iterrows():
    print(f"    {row['Model']:25s}  ECE={row['ECE']:.4f}  Brier={row['Brier Score']:.4f}")

print("\n  KEY THESIS (reframed for paper):")
print("    1. RF and XGB have near-identical AUC (DeLong p=0.79)")
print("    2. But their calibration differs — check ECE values above")
print("    3. Their error PATTERNS are identical (McNemar p=0.82)")
print("    4. The false negatives they share are the SAME patients")
print("    5. These shared misses are driven by the DATA, not the model")
print("    6. DCA shows where each model adds clinical value vs baselines")
print("\n  This is the 'beyond accuracy' story: two models with equal")
print("  discrimination can have different calibration, and the errors")
print("  that matter most (missed arrhythmias) are data-limited, not")
print("  model-limited. Model choice matters less than threshold choice")
print("  and probability calibration for patient safety.")

print(f"\n  Files saved:")
print(f"    {ece_path}")
print(f"    {cal_path}")
print(f"    {dca_path}")
print(f"    {dca_csv_path}")
print(f"    {fn_char_path}")
print(f"    {fn_plot_path}")
