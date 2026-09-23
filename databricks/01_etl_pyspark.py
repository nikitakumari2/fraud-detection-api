# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest and feature engineering with PySpark
# MAGIC
# MAGIC Reads `creditcard.csv` from a Unity Catalog volume, writes a **bronze** Delta table
# MAGIC (raw data + quality checks) and a **silver** Delta table with the streaming features
# MAGIC (`Time_Diff`, `Amount_Rolling_Avg_5`) computed with Spark window functions.
# MAGIC
# MAGIC **Before running:** upload `creditcard.csv` to the volume (see README, Databricks section).

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("volume", "fraud")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
BRONZE = f"{CATALOG}.{SCHEMA}.fraud_bronze"
SILVER = f"{CATALOG}.{SCHEMA}.fraud_silver"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
display(dbutils.fs.ls(VOLUME_PATH))

# COMMAND ----------

# MAGIC %md ## Bronze: raw data with an explicit schema

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

schema = StructType(
    [StructField("Time", DoubleType())]
    + [StructField(f"V{i}", DoubleType()) for i in range(1, 29)]
    + [StructField("Amount", DoubleType()), StructField("Class", IntegerType())]
)

raw = (spark.read.option("header", True).schema(schema)
       .csv(f"{VOLUME_PATH}/creditcard.csv")
       # keeps file order as a tie-breaker for rows with the same Time
       .withColumn("row_id", F.monotonically_increasing_id()))

raw.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(BRONZE)
bronze = spark.table(BRONZE)
print(f"{BRONZE}: {bronze.count():,} rows")

# COMMAND ----------

# MAGIC %md ## Data quality checks

# COMMAND ----------

feature_cols = [c for c in bronze.columns if c not in ("row_id",)]
nulls = bronze.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in feature_cols])
null_total = sum(nulls.first().asDict().values())
dupes = bronze.count() - bronze.dropDuplicates(feature_cols).count()

print(f"Null values: {null_total}")
print(f"Duplicate rows: {dupes:,}")
assert null_total == 0, "Unexpected nulls in the source data"

display(bronze.groupBy("Class").count()
        .withColumn("pct", F.round(100 * F.col("count") / F.sum("count").over(Window.partitionBy()), 3)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver: streaming features with window functions
# MAGIC
# MAGIC Same definitions as `app/features.py`, which the API uses at serving time:
# MAGIC - `Time_Diff`: seconds since the previous transaction (0 for the first)
# MAGIC - `Amount_Rolling_Avg_5`: mean `Amount` over the current and previous 4 transactions

# COMMAND ----------

w = Window.orderBy("Time", "row_id")

silver = (bronze
    .withColumn("Time_Diff", F.coalesce(F.col("Time") - F.lag("Time").over(w), F.lit(0.0)))
    .withColumn("Amount_Rolling_Avg_5", F.avg("Amount").over(w.rowsBetween(-4, 0))))

silver.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(SILVER)
print(f"{SILVER}: {spark.table(SILVER).count():,} rows")
display(spark.table(SILVER).orderBy("Time", "row_id").limit(10))

# COMMAND ----------

# MAGIC %md ## Quick analysis: fraud rate by hour of day

# COMMAND ----------

display(spark.table(SILVER)
    .withColumn("hour", (F.floor(F.col("Time") / 3600) % 24).cast("int"))
    .groupBy("hour")
    .agg(F.count("*").alias("transactions"),
         F.sum("Class").alias("frauds"),
         F.round(100 * F.avg("Class"), 3).alias("fraud_rate_pct"),
         F.round(F.avg("Amount"), 2).alias("avg_amount"))
    .orderBy("hour"))
