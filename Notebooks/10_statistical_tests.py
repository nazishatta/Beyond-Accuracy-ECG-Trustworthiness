"""
statistical_tests.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
Phase 2: Bootstrap confidence intervals, DeLong's test, McNemar's test.

Why this matters:
  - Point estimates (like AUC=0.96) tell you WHAT happened.
  - Confidence intervals tell you HOW CERTAIN you can be.
  - DeLong's test tells you if one model's AUC is TRULY better.
  - McNemar's test tells you if two models make DIFFERENT mistakes.
 
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, average_precision_score
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
DATA_DIR    = os.path.join(OUTPUT_DIR, "data")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 2 — STATISTICAL RIGOR")
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

# ================================================================
# PART 1: BOOTSTRAP 95% CONFIDENCE INTERVALS
# ================================================================
# How it works:
#   1. Take the test set (2198 patients).
#   2. Randomly pick 2198 patients WITH replacement (some patients
#      get picked twice, some not at all). This is one "resample."
#   3. Compute every metric (AUC, recall, etc.) on this resample.
#   4. Repeat 1000 times. Now you have 1000 AUC values.
#   5. Sort them. The 25th smallest = lower bound. The 975th = upper.
#      That range is your 95% confidence interval.
#   Why: if I repeated this study 100 times with different patients,
#   about 95 of those times the AUC would fall inside this range.
# ================================================================
print("\n" + "=" * 60)
print("PART 1: BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("=" * 60)

N_BOOTSTRAP = 1000
n_test = len(y_test)

def compute_metrics(y_true, y_pred, y_prob):
    """Compute all metrics for one bootstrap sample."""
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_true, y_prob),
        "Avg Prec":  average_precision_score(y_true, y_prob),
        "Brier":     brier_score_loss(y_true, y_prob),
    }

print(f"\nRunning {N_BOOTSTRAP} bootstrap resamples...")
boot_results = {name: [] for name in models}

for i in range(N_BOOTSTRAP):
    idx = np.random.choice(n_test, size=n_test, replace=True)
    yt  = y_test[idx]

    # Skip if resample has only one class (can't compute AUC)
    if len(np.unique(yt)) < 2:
        continue

    for name in models:
        yp = y_probs[name][idx]
        yd = y_preds[name][idx]
        boot_results[name].append(compute_metrics(yt, yd, yp))

print(f"Completed {len(boot_results['XGBoost'])} valid resamples.\n")

# Build summary table
ci_rows = []
for name in models:
    df_boot = pd.DataFrame(boot_results[name])
    row = {"Model": name}
    for metric in df_boot.columns:
        vals  = df_boot[metric].values
        point = compute_metrics(y_test, y_preds[name], y_probs[name])[metric]
        lo    = np.percentile(vals, 2.5)
        hi    = np.percentile(vals, 97.5)
        row[f"{metric}"]       = round(point, 4)
        row[f"{metric}_lo"]    = round(lo, 4)
        row[f"{metric}_hi"]    = round(hi, 4)
        row[f"{metric}_CI"]    = f"{point:.4f} ({lo:.4f}–{hi:.4f})"
    ci_rows.append(row)

ci_df = pd.DataFrame(ci_rows)

# Print nicely
for name in models:
    r = ci_df[ci_df["Model"] == name].iloc[0]
    print(f"  {name}:")
    for metric in ["AUC-ROC", "Recall", "Precision", "F1", "Brier", "Avg Prec", "Accuracy"]:
        print(f"    {metric:12s}: {r[f'{metric}_CI']}")
    print()

# Save
ci_path = os.path.join(RESULTS_DIR, "bootstrap_confidence_intervals.csv")
ci_df.to_csv(ci_path, index=False)
print(f"Saved: {ci_path}")

# ================================================================
# PART 2: DeLONG'S TEST (comparing two AUC values)
# ================================================================
# How it works:
#   Two models both have an AUC. Are they truly different, or could
#   the difference be random luck? DeLong's test answers this.
#   It gives a p-value:
#     p < 0.05 → the AUCs are significantly different (real gap)
#     p >= 0.05 → the difference could just be chance
#   Think of it like weighing two bags of rice. They look different,
#   but is the scale accurate enough to tell? DeLong checks the scale.
# ================================================================
print("\n" + "=" * 60)
print("PART 2: DeLONG'S TEST (AUC comparison)")
print("=" * 60)

def compute_midrank(x):
    """Compute midranks for DeLong's test."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n - 1 and z[k + 1] == z[k]:
            k += 1
        for t in range(i, k + 1):
            rank[t] = 0.5 * (i + k)
        i = k + 1
    rank2 = np.empty(n)
    rank2[j] = rank
    return rank2

def delong_test(y_true, prob_a, prob_b):
    """
    DeLong's test for comparing two AUCs.
    Returns: z-statistic, p-value, auc_a, auc_b
    """
    pos = y_true == 1
    neg = y_true == 0
    n1 = pos.sum()
    n0 = neg.sum()

    # Structural components for each model
    aucs = []
    v_list = []
    for prob in [prob_a, prob_b]:
        order = np.argsort(prob)
        ranks = compute_midrank(prob)

        # Placement values
        v10 = ranks[pos] / n0 - np.arange(1, n1 + 1) / n0 + 1.0 / (2.0 * n0)
        # Actually use a simpler formulation
        auc_val = roc_auc_score(y_true, prob)
        aucs.append(auc_val)

        # Compute placement values for variance estimation
        pos_probs = prob[pos]
        neg_probs = prob[neg]

        # For each positive sample, fraction of negatives it beats
        v10 = np.array([np.mean(p > neg_probs) + 0.5 * np.mean(p == neg_probs)
                        for p in pos_probs])
        # For each negative sample, fraction of positives that beat it
        v01 = np.array([np.mean(pos_probs > n) + 0.5 * np.mean(pos_probs == n)
                        for n in neg_probs])
        v_list.append((v10, v01))

    # Variance of the AUC difference
    s10_a, s01_a = v_list[0]
    s10_b, s01_b = v_list[1]

    var_a  = np.var(s10_a, ddof=1) / n1 + np.var(s01_a, ddof=1) / n0
    var_b  = np.var(s10_b, ddof=1) / n1 + np.var(s01_b, ddof=1) / n0
    cov_ab = (np.cov(s10_a, s10_b, ddof=1)[0, 1] / n1 +
              np.cov(s01_a, s01_b, ddof=1)[0, 1] / n0)

    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        return 0.0, 1.0, aucs[0], aucs[1]

    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p = 2.0 * stats.norm.sf(abs(z))  # two-sided
    return z, p, aucs[0], aucs[1]

# Compare all pairs
model_names = list(models.keys())
delong_rows = []
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        a, b = model_names[i], model_names[j]
        z, p, auc_a, auc_b = delong_test(y_test, y_probs[a], y_probs[b])
        sig = "YES (p<0.05)" if p < 0.05 else "NO (p>=0.05)"
        delong_rows.append({
            "Model A": a, "AUC A": round(auc_a, 4),
            "Model B": b, "AUC B": round(auc_b, 4),
            "Z-stat":  round(z, 4),
            "p-value": round(p, 6),
            "Significant": sig,
        })
        print(f"\n  {a} (AUC={auc_a:.4f}) vs {b} (AUC={auc_b:.4f})")
        print(f"    Z = {z:.4f}, p = {p:.6f}")
        print(f"    → {sig}")

delong_df = pd.DataFrame(delong_rows)
delong_path = os.path.join(RESULTS_DIR, "delong_test_results.csv")
delong_df.to_csv(delong_path, index=False)
print(f"\nSaved: {delong_path}")

# ================================================================
# PART 3: McNEMAR'S TEST (comparing error patterns)
# ================================================================
# How it works:
#   Two models look at the same 2198 patients. For each patient,
#   each model is either RIGHT or WRONG. We count four groups:
#     a = both RIGHT      b = Model A right, Model B wrong
#     c = Model A wrong, Model B right   d = both WRONG
#   McNemar's test checks: is b significantly different from c?
#   If yes, the models make genuinely different errors.
#   If no, they make basically the same mistakes.
#   Think of it like two students taking the same exam. If student A
#   gets questions 5,8,12 wrong and student B gets the same ones
#   wrong, they're not really different. McNemar checks this.
# ================================================================
print("\n" + "=" * 60)
print("PART 3: McNEMAR'S TEST (error pattern comparison)")
print("=" * 60)

def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test comparing two classifiers.
    Returns: chi2 statistic, p-value, contingency counts (b, c)
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # b = A correct, B wrong; c = A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    # McNemar's with continuity correction
    if b + c == 0:
        return 0.0, 1.0, int(b), int(c)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = stats.chi2.sf(chi2, df=1)
    return chi2, p, int(b), int(c)

mcnemar_rows = []
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        a, b = model_names[i], model_names[j]
        chi2, p, b_count, c_count = mcnemar_test(y_test, y_preds[a], y_preds[b])
        sig = "YES (p<0.05)" if p < 0.05 else "NO (p>=0.05)"
        mcnemar_rows.append({
            "Model A": a, "Model B": b,
            "A_right_B_wrong": b_count,
            "A_wrong_B_right": c_count,
            "Chi2":     round(chi2, 4),
            "p-value":  round(p, 6),
            "Significant": sig,
        })
        print(f"\n  {a} vs {b}")
        print(f"    {a} right, {b} wrong: {b_count}")
        print(f"    {a} wrong, {b} right: {c_count}")
        print(f"    Chi2 = {chi2:.4f}, p = {p:.6f}")
        print(f"    → {sig}")

mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_path = os.path.join(RESULTS_DIR, "mcnemar_test_results.csv")
mcnemar_df.to_csv(mcnemar_path, index=False)
print(f"\nSaved: {mcnemar_path}")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("PHASE 2 SUMMARY")
print("=" * 60)
print(f"\n  Bootstrap CIs:  {ci_path}")
print(f"  DeLong tests:   {delong_path}")
print(f"  McNemar tests:  {mcnemar_path}")
print(f"\n  All metrics now have 95% confidence intervals.")
print(f"  Every model comparison has a p-value.")

