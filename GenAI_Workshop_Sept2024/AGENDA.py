# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://companieslogo.com/img/orig/databricks_BIG-3be0f84a.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Generative AI Workshop with Databricks
# MAGIC
# MAGIC By the end of this course, you will have built an end-to-end Generative AI workflow that is ready for production!
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s): **15.4.x-cpu-ml-scala2.12, 15.4.x-gpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Admin Guide
# MAGIC
# MAGIC Please complete the following steps in a Databricks workspace for users to be able to run this lab:
# MAGIC * Grant permissions to the `main` UC Catalog in a new workspace: Catalog -> `main` -> Permissions -> users required `USE CATALOG`, `CREATE SCHEMA`, `CREATE TABLE`, and `CREATE PERMISSIONS` permissions. It is simplest to give `Data Editor` permission group to simplify this process. You can also edit the `init` notebook to point to a different catalog if needed.
# MAGIC * Enable Genie. Logged in as admin, click top-right -> Preview -> Genie `on`

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
