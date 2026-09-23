"""
Train the fraud model and save everything the API needs into model/.

Two ways to run it:
  - Locally:     python train.py      (needs creditcard.csv in this folder)
  - Databricks:  databricks/02_train_mlflow.py imports train_model() and save_artifacts()

Same approach as the original save_model.py (Time_Diff + rolling-average features,
SMOTE, RandomForest + XGBoost stacked with LogisticRegression), with three fixes:
  1. The scaler is fit on the training split only (no test-set leakage).
  2. The meta-model is trained on out-of-fold predictions, so it doesn't learn
     from base models that have already memorised the training rows.
  3. RandomForest depth is capped so the saved model fits in free-tier memory.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.features import SCALED_COLS, add_stream_features

DATA_FILE = Path("creditcard.csv")
RANDOM_STATE = 42
N_FOLDS = 3
TEST_SIZE = 0.3
THRESHOLD = 0.5
DEMO_ROWS = 3000  # held-out rows shipped with the API for the live demo

PARAMS = {
    "rf_n_estimators": 100, "rf_max_depth": 14, "rf_min_samples_leaf": 2,
    "xgb_n_estimators": 300, "xgb_max_depth": 6, "xgb_learning_rate": 0.1,
    "n_folds": N_FOLDS, "test_size": TEST_SIZE, "threshold": THRESHOLD,
    "resampling": "SMOTE", "meta_model": "LogisticRegression (out-of-fold)",
}


def base_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=PARAMS["rf_n_estimators"], max_depth=PARAMS["rf_max_depth"],
            min_samples_leaf=PARAMS["rf_min_samples_leaf"],
            random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=PARAMS["xgb_n_estimators"], max_depth=PARAMS["xgb_max_depth"],
            learning_rate=PARAMS["xgb_learning_rate"],
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    }


def train_model(df: pd.DataFrame) -> dict:
    """
    df: transactions sorted by Time, with Time_Diff and Amount_Rolling_Avg_5 already
    added (by add_stream_features locally, or by the Spark job on Databricks).
    """
    X = df.drop(columns="Class")
    y = df["Class"].astype(int)
    feature_order = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Fix 1: fit the scaler on training data only
    scaler = StandardScaler().fit(X_train[SCALED_COLS])
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[SCALED_COLS] = scaler.transform(X_train[SCALED_COLS])
    X_test_s[SCALED_COLS] = scaler.transform(X_test[SCALED_COLS])

    # Fix 2: out-of-fold base-model predictions for the meta-model.
    # SMOTE runs inside each fold so synthetic rows never leak into validation.
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = base_models()
    oof = np.zeros((len(X_train_s), len(models)))
    for i, (name, model) in enumerate(models.items()):
        print(f"Out-of-fold predictions: {name}...")
        pipe = ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("model", model)])
        oof[:, i] = cross_val_predict(pipe, X_train_s, y_train, cv=cv,
                                      method="predict_proba")[:, 1]
    meta = LogisticRegression().fit(oof, y_train)

    # Refit base models on the full SMOTE-resampled training set
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_s, y_train)
    fitted = {}
    for name, model in models.items():
        print(f"Final fit: {name}...")
        fitted[name] = model.fit(X_res, y_res)

    # Evaluate on the untouched test set
    test_base = np.column_stack([m.predict_proba(X_test_s)[:, 1] for m in fitted.values()])
    prob = meta.predict_proba(test_base)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    metrics = {
        "recall": float(recall_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, prob)),
    }
    print(classification_report(y_test, pred, target_names=["Non-Fraud", "Fraud"]))
    print(json.dumps(metrics, indent=2))

    bundle = {
        "base_models": fitted,
        "meta_model": meta,
        "scaler": scaler,
        "feature_order": feature_order,
        "threshold": THRESHOLD,
        "metrics": {**metrics, "threshold": THRESHOLD},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"bundle": bundle, "metrics": metrics, "df": df,
            "X_train": X_train, "X_test": X_test}


def save_artifacts(result: dict, out_dir) -> list:
    """Write the three files the API loads. Returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, X_train, X_test = result["df"], result["X_train"], result["X_test"]

    bundle_path = out_dir / "fraud_bundle.joblib"
    joblib.dump(result["bundle"], bundle_path, compress=3)

    # Real training rows (unscaled) for drift checks
    ref_path = out_dir / "reference_sample.csv.gz"
    X_train.sample(n=min(5000, len(X_train)), random_state=RANDOM_STATE).to_csv(
        ref_path, index=False, compression="gzip")

    # Demo stream: held-out rows in time order, every fraud case kept. Rows keep
    # the Time_Diff / rolling values they had in the full stream.
    test_rows = df.loc[X_test.index]
    frauds = test_rows[test_rows["Class"] == 1]
    n_legit = max(0, min(DEMO_ROWS - len(frauds), int((test_rows["Class"] == 0).sum())))
    legit = test_rows[test_rows["Class"] == 0].sample(n=n_legit, random_state=RANDOM_STATE)
    demo_path = out_dir / "demo_stream.csv.gz"
    pd.concat([frauds, legit]).sort_values("Time", kind="stable").to_csv(
        demo_path, index=False, compression="gzip")

    print(f"Saved {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB), "
          f"{ref_path.name}, {demo_path.name}")
    return [bundle_path, ref_path, demo_path]


def main():
    print("Loading data...")
    raw = pd.read_csv(DATA_FILE).sort_values("Time", kind="stable").reset_index(drop=True)
    df = add_stream_features(raw)
    result = train_model(df)
    save_artifacts(result, "model")


if __name__ == "__main__":
    main()
