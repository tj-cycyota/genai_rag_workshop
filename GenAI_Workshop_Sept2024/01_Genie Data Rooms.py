# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://companieslogo.com/img/orig/databricks_BIG-3be0f84a.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB 1. AI/BI Dashboards & Genie Spaces for Text-to-SQL GenAI
# MAGIC **Databricks GenAI Workshop**
# MAGIC
# MAGIC In this lab, we will explore Genie Data Rooms, a conversational experience for business teams to engage with their data through natural language. Genie leverages generative AI tailored to your organization’s business terminology and data and continuously learns from user feedback. This allows you to ask questions the same way you would ask an experienced coworker and receive relevant and accurate answers directly from your enterprise data.
# MAGIC
# MAGIC With Genie, users can self-serve and obtain answers to questions not addressed in their Dashboards without having to learn complicated tools or rely on expert practitioners. Backed by the lightning-fast performance and scale of Databricks SQL, users get answers immediately, all while upholding the governance and controls established in Unity Catalog.
# MAGIC
# MAGIC ![AIBI Dashboards](https://www.databricks.com/sites/default/files/inline-images/db-1001-blog-img-1.png?v=1718038955 "AIBI Dashboards")
# MAGIC
# MAGIC This lab will be mostly UI-focused, but we'll take a quick peak at our data. This very simple example is from Kaggle's [Laptop Price Prediction using specifications](https://www.kaggle.com/datasets/arnabchaki/laptop-price-prediction?resource=download&select=laptops_train.csv) (License: Database Contents License (DbCL) v1.0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s): **`15.4.x-cpu-ml-scala2.12, 15.4.x-gpu-ml-scala2.12`**
# MAGIC
# MAGIC You will also need access to Genie Data Spaces and a Serverless SQL Warehouse.

# COMMAND ----------

import os
filename = "Data/laptops_train.csv"
data_path = "file:" + os.path.join(os.getcwd(), filename)
print(data_path)  # See file in Data folder

data = spark.read.csv(data_path, header=True, inferSchema=True)

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC Pretty simple dataset about laptop specs! Start thinking about some analytics questions you would want to ask of it. 
# MAGIC
# MAGIC Now we'll write it out to a Delta Table to use for performant dashboarding, as if it were a production, live dataset constantly being updated. 

# COMMAND ----------

# Set UC variables, see init for more details
catalog_name = "main"
current_user = spark.sql("SELECT current_user() as username").collect()[0].username
schema_name = f'genai_workshop_{current_user.split("@")[0].replace(".","_")}'
table_name = "laptops"
print(f"Data will be saved to: {catalog_name}.{schema_name}.{table_name}")

# Save sample data to Delta
(
    data
    .toDF(*(col.replace(" ", "_") for col in data.columns))  # Get rid of spaces in column names
    .write.format("delta")
    .mode("overwrite")
    .saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a AI/BI Dashboard
# MAGIC
# MAGIC Follow the steps below to create a simple Dashboard visualization and Genie Data Room for this dataset:
# MAGIC 1. In the left-nav, open `Dashboards` in a new tab. Orient the tab so you can see both this current notebook and the new tab.
# MAGIC 2. At the top-right, click `Create Dashboard`. 
# MAGIC 3. Click to the `Data` tab at top. Click `Select a table`. Paste in the table name printed in the previous cell (e.g. `main.gen_ai_workshop...laptops`)
# MAGIC 4. After the query runs, click back to the `Canvas` tab. At the bottom, click the `Add a visualization` button, and drag the visualization onto the screen. 
# MAGIC 5. In the new box displaying "Ask the assistant to create a chart", type: `count of laptops by category`
# MAGIC 6. You can accept or reject the proposed chart, or use the chart editor at the right to make further edits. 
# MAGIC 7. Repeat this process until you have 3-4 visualizations displaying important analytics about this dataset. Get creative here! BONUS: can you figure out how to convert any of the columns saved as strings to numeric, such as `Screen_Size`?
# MAGIC 8. Once finished, click `Publish` at top-right, then view your published dashboard. 
# MAGIC
# MAGIC ![AIBI Dashboards](https://www.databricks.com/sites/default/files/inline-images/ai-bi-value-prop.gif?v=1717890485 "AIBI Dashboards")
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Link a Genie Space to your Dashboard
# MAGIC
# MAGIC Static dashboards are nice, but what if stakeholders have further questions about the data, or want to extend your analysis? This is a perfect time to use Genie, a text-to-sql interface that makes it easy to ask natural language questions of your data! 
# MAGIC
# MAGIC Let's add a Genie Space to our dashboard:
# MAGIC 1. In the editor mode of your dashboard (click `Draft` at top-center), click the top-right three vertical dots `...`
# MAGIC 2. Click `Create Genie Space`. Review the message and click `Yes`. A new tab will open with you Genie Space. 
# MAGIC 3. On the left side of the Space, click to the `Data` tab. You will see the `laptops` table you added as a data source for your dashboard. 
# MAGIC 4. Now click to `Settings`. Here you can edit properties of the Space, and add more tables if needed. Only tables added here will be queried by the AI model.
# MAGIC 5. Let's ask our first question! In the main Genie Space chat thread, type: `Which manufacturers offer non-Windows laptops?`. You should see a tabular response with the answer. Click the "Show generated code" button to verify the SQL query in use is as you would expect. Feel free to ask more questions or try the "Quick actions" at the bottom.
# MAGIC 6. That was an easy one, let's try something hard. Ask: `What is the average screen size?`. Now you will likely see the results as `null`. Click the generated code to see why. You can take this SQL and run it in the SQL Editor (at left-nav) yourself, but this is the wrong SQL function to process our messy data (e.g. every value in the table has an extra `""` after it.). We could clean it up, but let's have the LLM do it for us so our colleagues don't run into the same issue.
# MAGIC 6. Click to `Instructions` at the left side of the Genie Space. Here we can add "prompts" to guide the AI behind Genie. In the `General Instructions` box, add this instruction (and others if you like!):
# MAGIC     * `When performing any calculation on column "Screen_Size", always cast it to numeric with: CAST(REPLACE(Screen_Size, '"', '') AS DOUBLE)`
# MAGIC 7. Start a new chat (so it picks up the instruction) and ask the same question. You should see the correct answer now of ~15.05". Not only did the LLM use our data-specific instruction, but also applied our average calculation on top of it!
# MAGIC 8. Keep asking more questions of the dataset, adding instructions, example SQL queries, and maybe even a [UC SQL function](https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html). Start to brainstorm how you might turn your most valuable datasets into a "data product" with Genie Spaces. 
# MAGIC
# MAGIC ![Genie Space](https://www.databricks.com/sites/default/files/inline-images/intro-genie.gif?v=1717890485 "Genie Space")
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Have extra time?**
# MAGIC * Review the best practice guide on how to [Curate an effective Genie space](https://docs.databricks.com/en/genie/best-practices.html)
# MAGIC * Learn more: [What is an AI/BI Genie space?](https://docs.databricks.com/en/genie/index.html)
