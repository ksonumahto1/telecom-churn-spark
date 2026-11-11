# Telecom Customer Churn Prediction & Segmentation (PySpark on Databricks)

End-to-end data science workflow on Apache Spark using the IBM Telco Customer Churn dataset (7,043 customers, 21 attributes).

## What it does
1. **Data extraction** into a Spark DataFrame
2. **Data cleansing & quality checks**: null and duplicate checks, type casting, range checks
3. **Exploratory data analysis** with Spark SQL: churn by contract, internet service and tenure
4. **Feature engineering** with Spark ML pipelines: one-hot encoding, scaling, engineered features (service count, month-to-month flag, new-customer flag)
5. **Supervised learning**: Logistic Regression, Random Forest and Gradient-Boosted Trees, compared on ROC-AUC, F1, precision and recall
6. **Hyperparameter tuning**: 5-fold cross-validation over regularization parameters
7. **Model interpretation**: top churn drivers by feature importance
8. **Unsupervised learning**: K-Means segmentation, with k chosen by silhouette score

## Key results
- ROC-AUC of about 0.85 on the held-out test set
- Top churn drivers are month-to-month contracts, short tenure, no online security, fiber internet and no tech support
- K-Means found 3 segments; the month-to-month, short-tenure segment churns at about 44%, versus about 3% for long-term contract customers

## Run it on Databricks (free)
1. Sign up for Databricks Free Edition.
2. Go to Workspace, then Import, and upload `telecom_churn_pyspark.py`.
3. Attach serverless compute and click **Run all**. The data loads directly from the public URL.

## Tech
PySpark, Spark SQL, Spark MLlib, Databricks, pandas