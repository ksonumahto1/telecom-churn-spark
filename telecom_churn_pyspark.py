# Databricks notebook source
# MAGIC %md
# MAGIC # Telecom Customer Churn Prediction & Segmentation with PySpark
# MAGIC **Author:** Sonu Kumar
# MAGIC
# MAGIC End-to-end data science workflow on Spark (Databricks):
# MAGIC 1. Data extraction into a Spark DataFrame
# MAGIC 2. Data cleansing and statistical data quality checks
# MAGIC 3. Exploratory data analysis with Spark SQL
# MAGIC 4. Feature engineering with Spark ML pipelines
# MAGIC 5. Supervised learning: Logistic Regression, Random Forest, Gradient-Boosted Trees
# MAGIC 6. Hyperparameter tuning with cross-validation
# MAGIC 7. Model interpretation (feature importance)
# MAGIC 8. Unsupervised learning: K-Means customer segmentation
# MAGIC
# MAGIC **Data:** IBM Telco Customer Churn (7,043 customers, 21 attributes).

# COMMAND ----------

import os
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder.appName("TelecomChurn").getOrCreate()
SEED = 42

# COMMAND ----------

# MAGIC %md ## 1. Data extraction

# COMMAND ----------

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = os.environ.get("CHURN_DATA_PATH", DATA_URL)  # local path for testing, URL on Databricks

raw = spark.createDataFrame(pd.read_csv(DATA_PATH, dtype=str))
print(f"Rows: {raw.count():,} | Columns: {len(raw.columns)}")
raw.printSchema()

# COMMAND ----------

# MAGIC %md ## 2. Data cleansing and data quality checks

# COMMAND ----------

df = (raw
      .withColumn("tenure", F.col("tenure").cast("int"))
      .withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))
      .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
      .withColumn("TotalCharges", F.when(F.trim("TotalCharges") == "", None)
                                   .otherwise(F.col("TotalCharges")).cast("double"))
      .withColumn("label", (F.col("Churn") == "Yes").cast("int")))

# Data quality report: nulls and duplicate keys
nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
print("Null counts (non-zero):", {k: v for k, v in nulls.first().asDict().items() if v})
print("Duplicate customerIDs:", df.count() - df.select("customerID").distinct().count())

# Blank TotalCharges belong to brand-new customers (tenure = 0) -> no charges yet
df = df.withColumn("TotalCharges", F.coalesce("TotalCharges", F.lit(0.0)))

# Range checks
df.select(F.min("tenure"), F.max("tenure"), F.min("MonthlyCharges"), F.max("MonthlyCharges")).show()

# COMMAND ----------

# MAGIC %md ## 3. Exploratory data analysis (Spark SQL)

# COMMAND ----------

df.createOrReplaceTempView("customers")

print("Overall churn rate:")
spark.sql("SELECT ROUND(AVG(label)*100, 1) AS churn_rate_pct, COUNT(*) AS customers FROM customers").show()

print("Churn by contract type:")
spark.sql("""
    SELECT Contract, COUNT(*) AS customers, ROUND(AVG(label)*100, 1) AS churn_rate_pct,
           ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
    FROM customers GROUP BY Contract ORDER BY churn_rate_pct DESC
""").show()

print("Churn by internet service:")
spark.sql("""
    SELECT InternetService, COUNT(*) AS customers, ROUND(AVG(label)*100, 1) AS churn_rate_pct
    FROM customers GROUP BY InternetService ORDER BY churn_rate_pct DESC
""").show()

print("Churn by tenure band:")
spark.sql("""
    SELECT CASE WHEN tenure < 12 THEN '0-11 months' WHEN tenure < 24 THEN '12-23 months'
                WHEN tenure < 48 THEN '24-47 months' ELSE '48+ months' END AS tenure_band,
           COUNT(*) AS customers, ROUND(AVG(label)*100, 1) AS churn_rate_pct
    FROM customers GROUP BY 1 ORDER BY 1
""").show()

# COMMAND ----------

# MAGIC %md ## 4. Feature engineering

# COMMAND ----------

services = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

df = (df
      .withColumn("num_services", sum((F.col(c) == "Yes").cast("int") for c in services))
      .withColumn("avg_charge_per_month", F.when(F.col("tenure") > 0, F.col("TotalCharges") / F.col("tenure"))
                                           .otherwise(F.col("MonthlyCharges")))
      .withColumn("is_month_to_month", (F.col("Contract") == "Month-to-month").cast("int"))
      .withColumn("is_new_customer", (F.col("tenure") < 12).cast("int")))

categorical = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "num_services",
           "avg_charge_per_month", "is_month_to_month", "is_new_customer"]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical]
encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in categorical],
                        outputCols=[f"{c}_ohe" for c in categorical])
assembler = VectorAssembler(inputCols=[f"{c}_ohe" for c in categorical] + numeric, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features")
prep_stages = indexers + [encoder, assembler, scaler]

train, test = df.randomSplit([0.8, 0.2], seed=SEED)
print(f"Train: {train.count():,} | Test: {test.count():,}")

# COMMAND ----------

# MAGIC %md ## 5. Supervised learning: model selection

# COMMAND ----------

auc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
prec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="precisionByLabel", metricLabel=1.0)
rec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="recallByLabel", metricLabel=1.0)

models = {
    "Logistic Regression": LogisticRegression(labelCol="label", featuresCol="features", maxIter=100),
    "Random Forest": RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=200, seed=SEED),
    "Gradient-Boosted Trees": GBTClassifier(labelCol="label", featuresCol="features", maxIter=50, seed=SEED),
}

results, fitted = [], {}
for name, model in models.items():
    fit = Pipeline(stages=prep_stages + [model]).fit(train)
    pred = fit.transform(test)
    fitted[name] = fit
    results.append((name, auc_eval.evaluate(pred), f1_eval.evaluate(pred),
                    prec_eval.evaluate(pred), rec_eval.evaluate(pred)))

comparison = spark.createDataFrame(results, ["model", "roc_auc", "f1", "precision_churn", "recall_churn"])
comparison.select("model", *[F.round(c, 3).alias(c) for c in comparison.columns[1:]]).show(truncate=False)

# COMMAND ----------

# MAGIC %md ## 6. Hyperparameter tuning (5-fold cross-validation)

# COMMAND ----------

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=100)
grid = (ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .build())
cv = CrossValidator(estimator=Pipeline(stages=prep_stages + [lr]), estimatorParamMaps=grid,
                    evaluator=auc_eval, numFolds=5, seed=SEED, parallelism=4)
cv_model = cv.fit(train)
best = cv_model.bestModel
best_lr = best.stages[-1]
tuned_pred = best.transform(test)
print(f"Best regParam={best_lr.getRegParam()}, elasticNetParam={best_lr.getElasticNetParam()}")
print(f"Tuned Logistic Regression test ROC-AUC: {auc_eval.evaluate(tuned_pred):.3f}")

# COMMAND ----------

# MAGIC %md ## 7. Model interpretation: top churn drivers

# COMMAND ----------

def feature_names(pipeline_model):
    attrs = pipeline_model.transform(test.limit(1)).schema["raw_features"].metadata["ml_attr"]["attrs"]
    names = {}
    for group in attrs.values():
        for a in group:
            names[a["idx"]] = a["name"]
    return [names[i] for i in sorted(names)]

rf_model = fitted["Random Forest"]
names = feature_names(rf_model)
importances = rf_model.stages[-1].featureImportances.toArray()
top = sorted(zip(names, importances), key=lambda x: -x[1])[:10]
print("Top 10 churn drivers (Random Forest feature importance):")
for n, v in top:
    print(f"  {n:45s} {v:.3f}")

# COMMAND ----------

# MAGIC %md ## 8. Unsupervised learning: K-Means customer segmentation

# COMMAND ----------

seg_cols = ["tenure", "MonthlyCharges", "num_services", "is_month_to_month"]
seg_prep = Pipeline(stages=[VectorAssembler(inputCols=seg_cols, outputCol="seg_raw"),
                            StandardScaler(inputCol="seg_raw", outputCol="seg_features", withMean=True)]).fit(df)
seg_df = seg_prep.transform(df)

sil = ClusteringEvaluator(featuresCol="seg_features", predictionCol="segment")
scores = {}
for k in range(2, 7):
    km = KMeans(featuresCol="seg_features", predictionCol="segment", k=k, seed=SEED).fit(seg_df)
    scores[k] = sil.evaluate(km.transform(seg_df))
    print(f"k={k}: silhouette={scores[k]:.3f}")

k_best = max(scores, key=scores.get)
k_final = k_best if k_best >= 3 else 4  # keep enough segments to be actionable
segments = KMeans(featuresCol="seg_features", predictionCol="segment", k=k_final, seed=SEED).fit(seg_df).transform(seg_df)

print(f"Segment profiles (k={k_final}):")
(segments.groupBy("segment")
 .agg(F.count("*").alias("customers"),
      F.round(F.avg("tenure"), 1).alias("avg_tenure"),
      F.round(F.avg("MonthlyCharges"), 2).alias("avg_monthly"),
      F.round(F.avg("num_services"), 1).alias("avg_services"),
      F.round(F.avg("is_month_to_month") * 100, 1).alias("pct_month_to_month"),
      F.round(F.avg("label") * 100, 1).alias("churn_rate_pct"))
 .orderBy(F.desc("churn_rate_pct"))
 .show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business recommendations
# MAGIC - Target the highest-churn segment (short tenure, month-to-month) with contract-upgrade offers.
# MAGIC - Bundle security / tech-support services, which are associated with lower churn.
# MAGIC - Use the tuned model's churn probability to prioritize retention outreach each month.
