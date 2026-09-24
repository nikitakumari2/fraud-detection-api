"""
Fraud Detection API

Endpoints
  GET  /health          model status and offline test metrics
  POST /predict         score one transaction (add ?explain=true for top SHAP drivers)
  POST /predict/batch   score several transactions in time order
  GET  /demo/next       replay the next held-out test transactions through the model
  GET  /monitoring      live stats: volume, flag rate, latency, recall on replayed data, drift
  POST /reset           clear stream state and restart the demo replay
Interactive docs at /docs
"""

import os
import time
from collections import deque
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, create_model
from scipy.stats import ks_2samp

from app.features import RAW_COLS, SCALED_COLS, StreamState

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
THRESHOLD_OVERRIDE = os.getenv("FRAUD_THRESHOLD")
DRIFT_WINDOW = 500
DRIFT_FEATURES = ["Amount", "Time_Diff", "V14", "V4", "V12", "V10", "V17", "V3"]

# ---------- load artifacts once at startup ----------
bundle = joblib.load(MODEL_DIR / "fraud_bundle.joblib")
BASE_MODELS = bundle["base_models"]
META = bundle["meta_model"]
SCALER = bundle["scaler"]
FEATURE_ORDER = bundle["feature_order"]
THRESHOLD = float(THRESHOLD_OVERRIDE) if THRESHOLD_OVERRIDE else float(bundle["threshold"])
XGB_BOOSTER = BASE_MODELS["xgboost"].get_booster()

REFERENCE = pd.read_csv(MODEL_DIR / "reference_sample.csv.gz")
DEMO = pd.read_csv(MODEL_DIR / "demo_stream.csv.gz")

# ---------- request schemas ----------
Transaction = create_model(
    "Transaction",
    **{c: (float, Field(..., description=f"{c} value")) for c in RAW_COLS},
)
Transaction.model_config["json_schema_extra"] = {
    "example": DEMO.iloc[0][RAW_COLS].round(6).to_dict()
}


class BatchRequest(BaseModel):
    transactions: list[Transaction]  # type: ignore[valid-type]


# ---------- runtime state ----------
stream = StreamState()
stats_lock = Lock()
recent_features = deque(maxlen=DRIFT_WINDOW)
latencies_ms = deque(maxlen=1000)
counters = {"scored": 0, "flagged": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
demo_cursor = 0

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Stacked RandomForest + XGBoost model with SMOTE, served with FastAPI.",
    version="1.0.0",
)


def _score(raw: dict, explain: bool = False, actual: int | None = None,
           precomputed: bool = False) -> dict:
    start = time.perf_counter()

    row = {c: float(raw[c]) for c in RAW_COLS}
    if precomputed:  # demo rows carry their real stream features from the full timeline
        row.update({c: float(raw[c]) for c in ("Time_Diff", "Amount_Rolling_Avg_5")})
    else:
        row.update(stream.update(row["Time"], row["Amount"]))
    unscaled = pd.DataFrame([row])[FEATURE_ORDER]

    X = unscaled.copy()
    X[SCALED_COLS] = SCALER.transform(unscaled[SCALED_COLS])

    base = {name: float(m.predict_proba(X)[0, 1]) for name, m in BASE_MODELS.items()}
    prob = float(META.predict_proba(np.array([list(base.values())]))[0, 1])
    is_fraud = prob >= THRESHOLD

    result = {
        "fraud_probability": round(prob, 6),
        "is_fraud": bool(is_fraud),
        "threshold": THRESHOLD,
        "base_model_scores": {k: round(v, 6) for k, v in base.items()},
    }

    if explain:
        # XGBoost computes exact TreeSHAP values natively, no shap package needed
        contribs = XGB_BOOSTER.predict(
            xgb.DMatrix(X, feature_names=FEATURE_ORDER), pred_contribs=True)[0][:-1]
        top = np.argsort(-np.abs(contribs))[:5]
        result["top_features"] = [
            {"feature": FEATURE_ORDER[i], "shap_value": round(float(contribs[i]), 4),
             "direction": "toward fraud" if contribs[i] > 0 else "toward legitimate"}
            for i in top
        ]

    elapsed = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(elapsed, 2)

    with stats_lock:
        latencies_ms.append(elapsed)
        recent_features.append(unscaled.iloc[0].to_dict())
        counters["scored"] += 1
        counters["flagged"] += int(is_fraud)
        if actual is not None:
            key = {(1, 1): "tp", (0, 1): "fp", (1, 0): "fn", (0, 0): "tn"}[(actual, int(is_fraud))]
            counters[key] += 1
    return result


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_trained_at": bundle["trained_at"],
            "threshold": THRESHOLD, "offline_test_metrics": bundle["metrics"],
            "features": len(FEATURE_ORDER)}


@app.post("/predict")
def predict(txn: Transaction, explain: bool = Query(False)):  # type: ignore[valid-type]
    return _score(txn.model_dump(), explain=explain)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if len(req.transactions) > 500:
        raise HTTPException(413, "Send at most 500 transactions per batch.")
    ordered = sorted((t.model_dump() for t in req.transactions), key=lambda t: t["Time"])
    return {"results": [_score(t) for t in ordered]}


@app.get("/demo/next")
def demo_next(n: int = Query(5, ge=1, le=100), explain: bool = Query(False)):
    """Replay held-out test transactions the model never trained on."""
    global demo_cursor
    out = []
    for _ in range(n):
        if demo_cursor >= len(DEMO):
            demo_cursor = 0
        row = DEMO.iloc[demo_cursor]
        demo_cursor += 1
        actual = int(row["Class"])
        res = _score(row.to_dict(), explain=explain, actual=actual, precomputed=True)
        res.update({"time": float(row["Time"]), "amount": float(row["Amount"]),
                    "actual_fraud": bool(actual)})
        out.append(res)
    return {"position": demo_cursor, "total": len(DEMO), "transactions": out}


@app.get("/monitoring")
def monitoring():
    with stats_lock:
        c = dict(counters)
        lat = np.array(latencies_ms) if latencies_ms else np.array([0.0])
        recent = pd.DataFrame(list(recent_features))

    labelled = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    perf = None
    if labelled:
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        perf = {"labelled_transactions": labelled,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "confusion": {k: c[k] for k in ("tp", "fp", "fn", "tn")}}

    drift = []
    if len(recent) >= 50:
        for f in DRIFT_FEATURES:
            stat, p = ks_2samp(REFERENCE[f], recent[f])
            drift.append({"feature": f, "ks_statistic": round(float(stat), 4),
                          "p_value": round(float(p), 6), "drift": bool(p < 0.01)})

    return {
        "scored": c["scored"],
        "flagged": c["flagged"],
        "flag_rate": round(c["flagged"] / c["scored"], 5) if c["scored"] else 0.0,
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                       "p95": round(float(np.percentile(lat, 95)), 2)},
        "performance_on_replay": perf,
        "drift": {"window": len(recent), "min_window": 50, "features": drift},
    }


@app.post("/reset")
def reset():
    global demo_cursor
    stream.reset()
    with stats_lock:
        demo_cursor = 0
        recent_features.clear()
        latencies_ms.clear()
        for k in counters:
            counters[k] = 0
    return {"status": "reset"}
