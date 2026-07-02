"""
train_models.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
Step 3: Train Logistic Regression, Random Forest, and XGBoost models.
"""

import os, time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR   = os.path.join(OUTPUT_DIR, "data")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("STEP 3 — MODEL TRAINING")
print("=" * 60)

# ── Load processed training data ──────────────────────────────
train = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
X_train = train.drop("label", axis=1)
y_train = train["label"]
print(f"Training set: {X_train.shape} | Class balance: {dict(y_train.value_counts().sort_index())}")

# ── Model definitions ─────────────────────────────────────────
models = {
    "logistic_regression": LogisticRegression(
        solver="lbfgs", max_iter=1000, C=1.0,
        random_state=RANDOM_SEED, n_jobs=-1
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, eval_metric="logloss",
        use_label_encoder=False
    ),
}

# ── Train and save ─────────────────────────────────────────────
for name, model in models.items():
    print(f"\nTraining {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Done in {elapsed:.1f}s — saved to {path}")

print("\n✅ Step 3 complete. All models saved to outputs/models/")

