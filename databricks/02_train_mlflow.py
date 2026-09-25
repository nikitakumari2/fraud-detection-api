# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Train, track with MLflow, register in Unity Catalog
# MAGIC
# MAGIC Trains the stacked ensemble on the silver table using the **same** `train.py` code as
# MAGIC the local workflow, logs parameters/metrics/artifacts to MLflow, registers the model in
# MAGIC Unity Catalog, and exports the files the FastAPI app needs to the volume.

# COMMAND ----------

# MAGIC %pip install -q scikit-learn==1.8.0 xgboost==3.4.1 imbalanced-learn==0.14.2 joblib==1.5.3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath("..")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("volume", "fraud")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")

SILVER = f"{CATALOG}.{SCHEMA}.fraud_silver"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_detector"
EXPORT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/model"

import joblib, numpy as np, pandas as pd, sklearn, xgboost, imblearn
print(f"Repo root: {REPO_ROOT}")
print(f"numpy {np.__version__} | pandas {pd.__version__} | scikit-learn {sklearn.__version__} | "
      f"xgboost {xgboost.__version__} | imbalanced-learn {imblearn.__version__}")

# COMMAND ----------

# MAGIC %md ## Load the silver table and check Spark features match the serving code

# COMMAND ----------

from app.features import add_stream_features
from train import PARAMS, save_artifacts, train_model

pdf = (spark.table(SILVER).toPandas()
       .sort_values(["Time", "row_id"], kind="stable")
       .drop(columns="row_id")
       .reset_index(drop=True))

# Training/serving parity: the API computes these features with app/features.py,
# so they must match what Spark produced.
check = add_stream_features(pdf.drop(columns=["Time_Diff", "Amount_Rolling_Avg_5"]))
for col in ["Time_Diff", "Amount_Rolling_Avg_5"]:
    assert np.allclose(check[col], pdf[col]), f"{col} differs between Spark and app/features.py"
print(f"Feature parity check passed on {len(pdf):,} rows")

# Same column order as the local pipeline (Time, V1..V28, Amount, Class, Time_Diff, Rolling)
pdf = pdf[check.columns.tolist()]

# COMMAND ----------

# MAGIC %md ## Train with MLflow tracking

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fraud-detection")

SCALED_COLS = ["Amount", "Time", "Time_Diff", "Amount_Rolling_Avg_5"]


class FraudModel(mlflow.pyfunc.PythonModel):
    """Wraps the saved bundle so MLflow / Model Serving can score raw feature rows."""

    def load_context(self, context):
        import joblib
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input):
        import numpy as np
        import pandas as pd
        b = self.bundle
        X = model_input[b["feature_order"]].astype(float).copy()
        X[SCALED_COLS] = b["scaler"].transform(X[SCALED_COLS])
        base = np.column_stack([m.predict_proba(X)[:, 1] for m in b["base_models"].values()])
        prob = b["meta_model"].predict_proba(base)[:, 1]
        return pd.DataFrame({"fraud_probability": prob,
                             "is_fraud": (prob >= b["threshold"]).astype(int)})


with mlflow.start_run(run_name="stacked_rf_xgb_smote") as run:
    mlflow.log_params(PARAMS)
    mlflow.log_param("training_rows", len(pdf))
    mlflow.log_param("fraud_rate", round(float(pdf["Class"].mean()), 5))

    result = train_model(pdf)
    mlflow.log_metrics(result["metrics"])

    local_dir = "/tmp/fraud_model"
    paths = save_artifacts(result, local_dir)
    mlflow.log_artifacts(local_dir, artifact_path="api_model")

    example = result["X_test"].head(5)
    wrapper = FraudModel()
    wrapper.bundle = result["bundle"]
    signature = infer_signature(example, wrapper.predict(None, example))

    model_info = mlflow.pyfunc.log_model(
        artifact_path="fraud_model",
        python_model=FraudModel(),
        artifacts={"bundle": str(paths[0])},
        signature=signature,
        input_example=example,
        registered_model_name=MODEL_NAME,
        pip_requirements=[
            f"scikit-learn=={sklearn.__version__}",
            f"xgboost=={xgboost.__version__}",
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
            f"joblib=={joblib.__version__}",
        ],
    )

print(f"Run: {run.info.run_id}")
print(f"Registered {MODEL_NAME} version {model_info.registered_model_version}")

# COMMAND ----------

# MAGIC %md ## Mark this version as the champion and test loading it back

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "champion", model_info.registered_model_version)

loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
display(loaded.predict(result["X_test"].head(10)))

# COMMAND ----------

# MAGIC %md ## Export the API files to the volume (download these for the FastAPI app)

# COMMAND ----------

import shutil

os.makedirs(EXPORT_DIR, exist_ok=True)
for p in paths:
    shutil.copy(p, f"{EXPORT_DIR}/{os.path.basename(p)}")
display(dbutils.fs.ls(EXPORT_DIR))
