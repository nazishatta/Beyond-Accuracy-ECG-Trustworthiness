"""
cross_fold_validation.py
 ECG Trustworthiness Framework - Nazish Atta, George Washington University
 Phase 4: K-fold cross-validation across PTB-XL's 10 stratified folds.

Why this matters:
  Our main results used folds 1-8 for training and fold 10 for testing.
  But what if fold 10 happened to be "easy"? Would we get similar results
  on a different test fold?

  Cross-fold validation answers this by rotating: each of the 10 folds
  takes a turn as the test set, and the other 9 are used for training.
  If the AUC is consistently ~0.96 across all 10 rounds, the result
  is robust. If it swings wildly (0.80 one round, 0.99 the next),
  the result depends on luck.

  This is the standard way to show stability when a second independent
  dataset is not available (which is our case - 12SL features are
  specific to PTB-XL+).
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    accuracy_score, brier_score_loss, average_precision_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR,
    "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

print("=" * 60)
print("PHASE 4 - CROSS-FOLD VALIDATION (10 FOLDS)")
print("=" * 60)

# ── Load and prepare data (same logic as 01_load_data.py) ──────
meta = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_database.csv"),
                    index_col="ecg_id")
features = pd.read_csv(os.path.join(DATA_DIR, "features", "12sl_features.csv"))
features = features.set_index("ecg_id")
df = meta.join(features, how="inner")

# Rhythm-only label (same as Phase 1 fix)
scp_stmt = pd.read_csv(os.path.join(DATA_DIR, "scp_statements.csv"),
                        index_col=0)
RHYTHM_CODES  = set(scp_stmt.index[scp_stmt["rhythm"] == 1.0])
NORMAL_RHYTHM = {"SR", "STACH", "SBRAD", "SARRH"}

def is_arrhythmia(scp_str):
    try:
        codes = set(ast.literal_eval(scp_str).keys())
        return 1 if (codes & RHYTHM_CODES) - NORMAL_RHYTHM else 0
    except Exception:
        return 0

df["label"] = df["scp_codes"].apply(is_arrhythmia)
feature_cols = [c for c in features.columns if c in df.columns]
print(f"Dataset: {len(df)} records | {len(feature_cols)} features | "
      f"Arrhythmia: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")

# ── Model definitions (same as 03_train_models.py, no double correction) ──
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1.0,
            random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, eval_metric="logloss",
            use_label_encoder=False
        ),
    }

# ── Run 10-fold cross-validation ───────────────────────────────
folds = sorted(df["strat_fold"].unique())
print(f"Folds: {folds}")
print(f"\nRunning 10-fold cross-validation (this takes a few minutes)...\n")

all_results = []

for test_fold in folds:
    train_mask = df["strat_fold"] != test_fold
    test_mask  = df["strat_fold"] == test_fold

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "label"].values
    X_test  = df.loc[test_mask,  feature_cols].values
    y_test  = df.loc[test_mask,  "label"].values

    # Impute, scale, SMOTE (same pipeline as 02_preprocess.py)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Train and evaluate each model
    models = get_models()
    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        result = {
            "Fold":      int(test_fold),
            "Model":     name,
            "N_test":    len(y_test),
            "N_arr":     int(y_test.sum()),
            "Prevalence": round(y_test.mean(), 4),
            "AUC-ROC":   round(roc_auc_score(y_test, y_prob), 4),
            "Avg_Prec":  round(average_precision_score(y_test, y_prob), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Brier":     round(brier_score_loss(y_test, y_prob), 4),
            "FN":        int(((y_test == 1) & (y_pred == 0)).sum()),
            "FP":        int(((y_test == 0) & (y_pred == 1)).sum()),
        }
        all_results.append(result)

    print(f"  Fold {int(test_fold):2d} done  |  "
          f"Test n={len(y_test)}, arr={y_test.sum()}  |  "
          f"RF AUC={[r for r in all_results if r['Fold']==test_fold and r['Model']=='Random Forest'][0]['AUC-ROC']:.4f}  "
          f"XGB AUC={[r for r in all_results if r['Fold']==test_fold and r['Model']=='XGBoost'][0]['AUC-ROC']:.4f}")

# ── Build summary ──────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_path = os.path.join(RESULTS_DIR, "cross_fold_all_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nSaved: {results_path}")

print("\n" + "=" * 60)
print("CROSS-FOLD SUMMARY (mean ± std across 10 folds)")
print("=" * 60)

summary_rows = []
for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
    subset = results_df[results_df["Model"] == name]
    row = {"Model": name}
    print(f"\n  {name}:")
    for metric in ["AUC-ROC", "Recall", "Precision", "F1", "Brier", "Accuracy"]:
        vals = subset[metric].values
        mean_val = vals.mean()
        std_val  = vals.std()
        row[f"{metric}_mean"] = round(mean_val, 4)
        row[f"{metric}_std"]  = round(std_val, 4)
        row[f"{metric}_summary"] = f"{mean_val:.4f} ± {std_val:.4f}"
        print(f"    {metric:12s}: {mean_val:.4f} ± {std_val:.4f}  "
              f"(range: {vals.min():.4f}–{vals.max():.4f})")
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(RESULTS_DIR, "cross_fold_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")

# ── Plot: AUC across folds ────────────────────────────────────
COLORS = {"Logistic Regression": "#4472C4",
          "Random Forest": "#ED7D31",
          "XGBoost": "#70AD47"}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: AUC per fold
ax = axes[0]
for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
    subset = results_df[results_df["Model"] == name]
    ax.plot(subset["Fold"].values, subset["AUC-ROC"].values,
            "o-", color=COLORS[name], lw=2, ms=7, label=name)
ax.set_xlabel("Test Fold", fontsize=12)
ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title("AUC-ROC Stability Across 10 Folds\n"
             "Flat lines = robust results",
             fontsize=13, fontweight="bold")
ax.set_xticks(folds)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: Recall per fold
ax = axes[1]
for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
    subset = results_df[results_df["Model"] == name]
    ax.plot(subset["Fold"].values, subset["Recall"].values,
            "s-", color=COLORS[name], lw=2, ms=7, label=name)
ax.set_xlabel("Test Fold", fontsize=12)
ax.set_ylabel("Recall (Sensitivity)", fontsize=12)
ax.set_title("Recall Stability Across 10 Folds\n"
             "How consistently does each model catch arrhythmias?",
             fontsize=13, fontweight="bold")
ax.set_xticks(folds)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "cross_fold_stability.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {plot_path}")

# ── Stability assessment ───────────────────────────────────────
print("\n" + "=" * 60)
print("STABILITY ASSESSMENT")
print("=" * 60)

for name in ["Random Forest", "XGBoost"]:
    subset = results_df[results_df["Model"] == name]
    auc_std = subset["AUC-ROC"].std()
    rec_std = subset["Recall"].std()
    if auc_std < 0.02:
        stability = "HIGHLY STABLE (std < 0.02)"
    elif auc_std < 0.04:
        stability = "STABLE (std < 0.04)"
    else:
        stability = "VARIABLE (std >= 0.04) - investigate"
    print(f"\n  {name}:")
    print(f"    AUC std = {auc_std:.4f} → {stability}")
    print(f"    Recall std = {rec_std:.4f}")

print("\n  LIMITATION (state in paper):")
print("    True external validation on an independent dataset was not")
print("    feasible because the 12SL features used in this study are")
print("    specific to PTB-XL+. Cross-fold validation demonstrates")
print("    robustness across patient subsets within the same cohort.")
print("    External validation using raw-signal models or alternative")
print("    feature sets is recommended as future work.")


