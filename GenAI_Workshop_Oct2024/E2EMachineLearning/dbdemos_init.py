# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://companieslogo.com/img/orig/databricks_BIG-3be0f84a.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC Run the code below to initialize a Databricks Tutorial for [MLOps — End-to-End Pipeline](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/mlops-end-to-end-pipeline?itm_data=demo_center)
# MAGIC
# MAGIC Once the content installs, start with the `00_` notebook and follow the instructions.

# COMMAND ----------

# MAGIC %pip install -qq dbdemos 

# COMMAND ----------

# Setup Catalog+Schema
dbutils.widgets.text("catalog_name","main")

catalog_name = dbutils.widgets.get("catalog_name")

current_user = spark.sql("SELECT current_user() as username").collect()[0].username
current_user_safe = current_user.split("@")[0].replace(".","_")
schema_name = f'genai_workshop_{current_user.split("@")[0].replace(".","_")}'

_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
_ = spark.sql(f"USE {catalog_name}.{schema_name}")

print(f"Using catalog+schema: {catalog_name}.{schema_name}")

# COMMAND ----------

import dbdemos
dbdemos.install(
  'mlops-end2end',
  use_current_cluster=True,
  catalog = catalog_name,
  schema = schema_name
  )

# COMMAND ----------


