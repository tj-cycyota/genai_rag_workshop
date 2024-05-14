# Databricks notebook source
# MAGIC %md
# MAGIC This notebook based on open-code example located here: https://github.com/stikkireddy/composable-chainlit-ui-demo.git
# MAGIC
# MAGIC **NOTICE**: Before running this notebook, go into the `config.yaml` file and edit these parts:
# MAGIC ```
# MAGIC # REPLACE WITH YOUR ENDPOINT NAME
# MAGIC vector_search_endpoint_name: vs_endpoint_genai_workshop_YOURLABUSERNAME
# MAGIC
# MAGIC # REPLACE WITH YOUR INDEX NAME
# MAGIC vector_search_index_name: main.genai_workshop_YOURLABUSERNAME.product_index_YOURLABUSERNAME
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %pip install -q -U dbtunnel[chainlit,asgiproxy] chainlit==1.0.400
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from dbtunnel import dbtunnel

current_directory = os.getcwd()
script_path = current_directory + "/chatbot.py"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Host chatbot locally from your VM, only single user has access to UI

# COMMAND ----------

dbtunnel.chainlit(script_path).inject_auth().run()

# COMMAND ----------

# MAGIC %md
# MAGIC Search the cell above for `Use this link to access` to get the unique URL where your chatbot UI is hosted. Open that link in a new browser tab for a simple chatbot UI to test your chatbot and see how the chain is executed. Try asking some of the questions we used in earlier parts of our lab to test the performance of the chatbot, such as `What is the expected accuracy of the thermocouples in the sensor?`

# COMMAND ----------

# MAGIC %md
# MAGIC
