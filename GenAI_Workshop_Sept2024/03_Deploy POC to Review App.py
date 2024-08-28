# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://companieslogo.com/img/orig/databricks_BIG-3be0f84a.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB 3. Deploy POC to Review App
# MAGIC **Databricks GenAI Workshop**
# MAGIC
# MAGIC TODO

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s): **`15.4.x-cpu-ml-scala2.12, 15.4.x-gpu-ml-scala2.12`**

# COMMAND ----------

# MAGIC %md
# MAGIC First, run this pip install and initialization script to set a few parameters:

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

import os, mlflow, time
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review our chain logic
# MAGIC
# MAGIC Open the Python file in the same directory with the file name below. This contains the `LangChain` logic for our POC RAG application.

# COMMAND ----------

CHAIN_CODE_FILE = "multi_turn_rag_chain"

# COMMAND ----------

# MAGIC %md ## Log the chain to MLflow & test the RAG chain locally
# MAGIC
# MAGIC This will save the chain using MLflow's code-based logging and invoke it locally to test it.  
# MAGIC
# MAGIC **MLflow Tracing** allows you to inspect what happens inside the chain.  This same tracing data will be logged from your deployed chain along with feedback that your stakeholders provide to a Delta Table.

# COMMAND ----------

# Log the model to MLflow
with mlflow.start_run(run_name="poc_"+current_user_safe):
    # Tag to differentiate from the data pipeline runs
    mlflow.set_tag("type", "chain")

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), CHAIN_CODE_FILE
        ),  # Chain code file e.g., /path/to/the/chain.py
        model_config=rag_chain_config,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example=rag_chain_config[
            "input_example"
        ],  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        extra_pip_requirements=["databricks-agents"]
    )

    # # Attach the data pipeline's configuration as parameters
    # mlflow.log_params(_flatten_nested_params({"data_pipeline": data_pipeline_config}))

    # # Attach the data pipeline configuration 
    # mlflow.log_dict(data_pipeline_config, "data_pipeline_config.json")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the chain locally

# COMMAND ----------

chain_input = {
    "messages": [
        {
            "role": "user",
            "content": "How do I scan a document?", # Replace with a question relevant to your use case
        }
    ]
}
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(chain_input)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to the Review App
# MAGIC
# MAGIC Now, let's deploy the POC to the Review App so your stakeholders can provide you feedback.
# MAGIC
# MAGIC Notice how simple it is to call `agents.deploy()` to enable the Review App and create an API endpoint for the RAG chain!

# COMMAND ----------

instructions_to_reviewer = f"""## Instructions for Testing this RAG Initial Proof of Concept (PoC)

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Documents**:
   - Carefully review each document that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing. Your contributions are essential to delivering a high-quality product to our end users."""

print(instructions_to_reviewer)

# COMMAND ----------

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=rag_model_name)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=rag_model_name, model_version=uc_registered_model_info.version)

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

# Add the user-facing instructions to the Review App
agents.set_review_instructions(rag_model_name, instructions_to_reviewer)

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

print(f"\n\nReview App: {deployment_info.review_app_url}")
