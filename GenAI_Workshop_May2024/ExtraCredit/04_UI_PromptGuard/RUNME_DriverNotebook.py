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
# MAGIC Search the cell above for `Use this link to access` to get the unique URL where your chatbot UI is hosted.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Host application publicly on the internet (proxy server in backend)

# COMMAND ----------

# (
#   dbtunnel.chainlit(script_path, cwd=current_directory)
#   .inject_auth()
# .share_to_internet(
#   app_name="databricks-demo",
#   app_host="dbtunnel.app",
#   tunnel_host="proxy.dbtunnel.app"
# ).run())

# COMMAND ----------

# MAGIC %md
# MAGIC
