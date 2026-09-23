# Credit Card Fraud Detection API

A real-time fraud scoring service built with **Databricks**, **FastAPI** and **Docker**. Data is ingested and feature-engineered with PySpark into Delta tables, the model is trained and tracked with MLflow and registered in Unity Catalog, and the exported model is served by a containerised FastAPI app. A stacked ensemble (Random Forest + XGBoost with a Logistic Regression meta-model, trained with SMOTE) scores transactions from the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and returns a fraud probability with SHAP-based explanations.

**Live demo:** `https://<your-service>.onrender.com/docs`
*(Hosted on a free tier, so the first request after idle can take about a minute to wake up.)*

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Model status and offline test-set metrics |
| POST | `/predict` | Score one transaction. Add `?explain=true` for the top 5 SHAP drivers |
| POST | `/predict/batch` | Score up to 500 transactions in time order |
| GET | `/demo/next?n=5` | Replay held-out test transactions the model never saw, with true labels |
| GET | `/monitoring` | Volume, flag rate, p50/p95 latency, recall/precision on replayed data, KS-test drift |
| POST | `/reset` | Clear stream state and restart the replay |

## How it works

1. **Feature engineering** (`app/features.py`): `Time_Diff` and a 5-transaction rolling average of `Amount`. The same module is used for training and serving, so features can't silently diverge. At serving time a small in-memory buffer computes them for each incoming transaction.
2. **Model** (`train.py`): scaler fit on the training split only, SMOTE applied inside each CV fold, Random Forest and XGBoost base models, Logistic Regression meta-model trained on out-of-fold predictions.
3. **Explainability**: XGBoost's native TreeSHAP (`pred_contribs=True`), which avoids the heavy `shap` package and keeps the image small.
4. **Monitoring**: Kolmogorov–Smirnov tests compare the last 500 scored transactions against a sample of real training rows.

## Pipeline

```
creditcard.csv → [Databricks] PySpark → Delta bronze/silver → MLflow training → Unity Catalog model
                                                                     ↓ export
                                               [Render] Docker → FastAPI → /predict, /demo, /monitoring
```

## Run it

### 1a. Train on Databricks 
1. In Databricks Free Edition, open **Catalog → workspace → default → Create → Volume**, name it `fraud`, and upload `creditcard.csv` to it.
2. **Workspace → Create → Git folder**, paste this repo's URL.
3. Run `databricks/01_etl_pyspark.py`, then `databricks/02_train_mlflow.py`.
4. Download the three files from `/Volumes/workspace/default/fraud/model/` into this repo's `model/` folder.

### 1b. Or train locally
Download `creditcard.csv` from Kaggle into this folder, then:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt
python train.py
pytest
```
This writes `model/fraud_bundle.joblib`, `model/reference_sample.csv.gz` and `model/demo_stream.csv.gz`. Commit these; the CSV itself is git-ignored (it's over GitHub's 100 MB limit).

### 2. Run the API locally
```bash
uvicorn app.main:app --reload
```
Open http://localhost:8000/docs and try `/demo/next` or `/predict` (it comes with a prefilled example).

### 3. Run with Docker
```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

### 4. Deploy to Render (free)
1. Push this folder to a GitHub repo, including `model/`.
2. On render.com: **New → Web Service**, connect the repo. Render detects the Dockerfile.
3. Pick the **Free** instance type, set **Health Check Path** to `/health`, and deploy.

Optional: set a `FRAUD_THRESHOLD` environment variable to change the decision threshold without retraining.

## Example
```bash
curl "https://<your-service>.onrender.com/demo/next?n=3&explain=true"
```

## Notes and limitations
- The dataset has no card or customer ID, so `Time_Diff` and the rolling average describe the overall transaction stream, not per-card behaviour. With real data these would be computed per card.
- Stream state and monitoring counters live in memory, so the service runs a single worker and resets on restart. A production version would keep them in Redis or a feature store.
- V1–V28 are PCA components from the original dataset, so SHAP explanations point to anonymised features.
