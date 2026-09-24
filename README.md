# Credit Card Fraud Detection API

An end-to-end fraud detection pipeline: **PySpark ETL on Databricks**, a stacked **Random Forest + XGBoost** ensemble trained with **SMOTE** and tracked in **MLflow**, and a containerized **FastAPI** service deployed on **Render** with SHAP explanations and live drift monitoring.

**Live demo:** https://fraud-detection-api-i57g.onrender.com
*(Free tier: the service sleeps after ~15 minutes idle, so the first request can take up to a minute to wake it.)*

## Results

Evaluated on a held-out test set of 85,443 transactions (148 fraud), which the model never saw during training:

| Metric | Score |
|---|---|
| Recall | 77.7% |
| Precision | 88.5% |
| MCC | 0.829 |
| PR-AUC | 0.831 |

Accuracy isn't reported because with a 0.17% fraud rate, predicting "not fraud" for everything scores 99.8%. Recall, precision, MCC and PR-AUC reflect performance on the class that matters.

## Architecture

```
creditcard.csv (Kaggle, 284,807 transactions)
        │
        ▼
┌─────────────────────── Databricks ───────────────────────┐
│ 01_etl_pyspark     CSV → Delta bronze → quality checks   │
│                    → window-function features → silver │
│ 02_train_mlflow    SMOTE + stacked RF/XGBoost            │
│                    → MLflow tracking → Unity Catalog     │
│                    → export model files                  │
└──────────────────────────────────────────────────────────┘
        │  fraud_bundle.joblib, reference_sample, demo_stream
        ▼
┌──────────────── Docker container on Render ──────────────┐
│ FastAPI: /predict · /demo/next · /monitoring · /health  │
└──────────────────────────────────────────────────────────┘
```

## What's inside

**Data engineering (PySpark, Delta Lake).** Raw data lands in a bronze Delta table with an explicit schema and quality checks (nulls, duplicates, class balance). A silver table adds streaming features computed with Spark window functions: `Time_Diff` (seconds since the previous transaction) and `Amount_Rolling_Avg_5` (mean amount over the last 5 transactions).

**Modeling.** Random Forest and XGBoost base models are trained on SMOTE-resampled data, and a Logistic Regression meta-model combines them. To avoid data leakage:
- the scaler is fit on the training split only,
- SMOTE is applied inside each cross-validation fold,
- the meta-model is trained on out-of-fold predictions.

**Experiment tracking.** Parameters, metrics and artifacts are logged to MLflow. The model is registered in Unity Catalog as `workspace.default.fraud_detector` with a `champion` alias.

**Training/serving parity.** The same feature code (`app/features.py`) runs in training and in the API, and the Databricks notebook asserts that Spark's features exactly match it before training.

**Serving.** FastAPI with request validation, SHAP explanations using XGBoost's native TreeSHAP, and a monitoring endpoint that runs Kolmogorov–Smirnov drift tests against real training data. It's packaged in a slim Docker image running as a non-root user with a health check.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/docs` | Interactive API documentation (the root URL redirects here) |
| GET | `/health` | Model status, training date and test-set metrics |
| POST | `/predict` | Score one transaction. Add `?explain=true` for the top 5 SHAP drivers |
| POST | `/predict/batch` | Score up to 500 transactions in time order |
| GET | `/demo/next?n=50` | Replay held-out test transactions with their true labels |
| GET | `/monitoring` | Volume, flag rate, p50/p95 latency, live recall/precision, drift tests |
| POST | `/reset` | Reset stream state, counters and the demo replay |

**Try it:** open the live demo, run `/demo/next` with `n=100` a few times, then open `/monitoring` to see recall, precision and drift update.

Example:
```bash
curl "https://fraud-detection-api-i57g.onrender.com/demo/next?n=3&explain=true"
```

## Project structure

```
├── app/
│   ├── main.py              FastAPI application
│   └── features.py          Feature engineering shared by training and serving
├── databricks/
│   ├── 01_etl_pyspark.py    Bronze/silver Delta tables with PySpark
│   └── 02_train_mlflow.py   Training, MLflow tracking, Unity Catalog registration
├── model/                   Trained model files (exported from Databricks)
├── tests/test_api.py        API tests
├── train.py                 Training pipeline (used by Databricks and locally)
├── Dockerfile
├── requirements.txt         Serving dependencies
└── requirements-train.txt   Training and test dependencies
```

## Run it yourself

### 1. Train on Databricks
1. In Databricks Free Edition, go to **Catalog → workspace → default → Create → Volume**, name it `fraud`, and upload `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Go to **Workspace → Create → Git folder** and paste this repo's URL.
3. On serverless compute, run `databricks/01_etl_pyspark.py`, then `databricks/02_train_mlflow.py`.
4. Download the three files from `/Volumes/workspace/default/fraud/model/` into `model/`.

To train locally instead, put `creditcard.csv` in the project folder and run `python train.py`.

### 2. Run locally
Requires **Python 3.12+**.
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
python -m pytest
python -m uvicorn app.main:app --reload
```
Open http://localhost:8000.

### 3. Run with Docker
```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

### 4. Deploy to Render
Push to GitHub (including `model/`), then create a **Web Service** from the repo on Render. It detects the Dockerfile. Choose the Free instance and set the Health Check Path to `/health`.

Library versions in `requirements.txt` are pinned to match the Databricks training environment, so the saved model loads identically everywhere.

## Limitations and next steps

- **No card ID in the dataset.** `Time_Diff` and the rolling average describe the overall transaction stream, not individual customers. With real data, these would be computed per card using `Window.partitionBy("card_id")`, which would also let Spark distribute the work.
- **Duplicates.** The dataset contains 1,081 duplicate rows, found during quality checks. They were kept to preserve the stream order the rolling features depend on, at the cost of a small optimistic bias in test scores.
- **In-memory state.** Stream state and monitoring counters live in memory, so the service runs a single worker and resets on restart. A production version would use Redis or a feature store.
- **Threshold.** The decision threshold is 0.5. It can be changed with the `FRAUD_THRESHOLD` environment variable; a production system would tune it on validation data based on the cost of missed fraud versus false alarms.
- **Anonymized features.** V1–V28 are PCA components, so SHAP explanations point to anonymized features.

## Tech stack

Databricks · PySpark · Delta Lake · Unity Catalog · MLflow · scikit-learn · XGBoost · imbalanced-learn · FastAPI · Docker · Render