# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://companieslogo.com/img/orig/databricks_BIG-3be0f84a.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB 1. Genie Data Rooms for Text-to-SQL GenAI
# MAGIC **Databricks GenAI Workshop**
# MAGIC
# MAGIC In this lab, we will explore Genie Data Rooms, a conversational experience for business teams to engage with their data through natural language. Genie leverages generative AI tailored to your organization’s business terminology and data and continuously learns from user feedback. This allows you to ask questions the same way you would ask an experienced coworker and receive relevant and accurate answers directly from your enterprise data.
# MAGIC
# MAGIC With Genie, users can self-serve and obtain answers to questions not addressed in their Dashboards without having to learn complicated tools or rely on expert practitioners. Backed by the lightning-fast performance and scale of Databricks SQL, users get answers immediately, all while upholding the governance and controls established in Unity Catalog.
# MAGIC
# MAGIC This lab will be mostly UI-focused, but we'll take a quick peak at our data. This very simple example is from Kaggle's [Laptop Price Prediction using specifications](https://www.kaggle.com/datasets/arnabchaki/laptop-price-prediction?resource=download&select=laptops_train.csv) (License: Database Contents License (DbCL) v1.0)

# COMMAND ----------

import os
filename = "Data/laptops_train.csv"
data_path = "file:" + os.path.join(os.getcwd(), filename)
print(data_path)

data = spark.read.csv(data_path, header=True, inferSchema=True)

display(data)

# COMMAND ----------


