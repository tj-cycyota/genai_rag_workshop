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
# MAGIC In this notebook we will deploy a fully functional RAG application to Model Serving, then interact with our application in the Agent Review UI. 
# MAGIC
# MAGIC This notebook extends the topics discussed in earlier notebooks, so make sure you run them first! We'll complete the following steps:
# MAGIC
# MAGIC 1. **Review** the **LangChain** application logic that will perform retrieval augmented generation in response to a user's chat message. 
# MAGIC
# MAGIC 2. **Log** the application code to **MLflow** with the supported LangChain model flavor. 
# MAGIC
# MAGIC 3. **Test** the chain locally by loading it back from MLflow. 
# MAGIC
# MAGIC 4. **Deploy** the chain to a real-time model serving endpoints and the Agent Review UI. 
# MAGIC
# MAGIC 5. **Interact** with the RAG chain via the Review UI and ask questions related to our document(s).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s): **`15.4.x-cpu-ml-scala2.12, 15.4.x-gpu-ml-scala2.12`**
# MAGIC
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

# MAGIC %md
# MAGIC Spend a few minutes reviewing the code and how it uses configuration values from the `rag_chain_config.yaml` file (also in the same directory). The point of this exercise is not to become an expert on this particular chain framework, but rather to notice **how to tie together various components into a composable Compound AI system**. 
# MAGIC
# MAGIC For further learning, we recommend the following articles:
# MAGIC * [Create and log AI agents](https://docs.databricks.com/en/generative-ai/create-log-agent.html)
# MAGIC * [LangChain on Databricks for LLM development](https://docs.databricks.com/en/large-language-models/langchain.html)
# MAGIC * [Building Enterprise GenAI Apps with Meta Llama 3 on Databricks](https://www.databricks.com/blog/building-enterprise-genai-apps-meta-llama-3-databricks)

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
        model_config=rag_chain_config,  # Chain configuration set in init
        artifact_path="chain",  # Required by MLflow
        input_example=rag_chain_config[
            "input_example"
        ],  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        extra_pip_requirements=["databricks-agents"]
    )

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

chain_output = chain.invoke(chain_input)
display(chain_output)

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

# COMMAND ----------

# MAGIC %md
# MAGIC That's it, your agent is deployed!
# MAGIC
# MAGIC Visit the 2 links printed out in the cell above. These are:
# MAGIC * Model Serving Endpoint hosting your RAG application. [Model Serving info](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
# MAGIC * Agent Review UI for subject matter experts to review the applications behavior and performance, as well as provide "ground truth" data for automated evaluation. [Agent Review UI info](https://docs.databricks.com/en/generative-ai/deploy-agent.html)
# MAGIC
# MAGIC Once the Model Serving endpoint is ready (should take ~10-15 mins), go to the chat UI and ask questions about your document. You can then provide feedback/edits, which will be logged to the Inference Table and used in later sections for evaluation.
# MAGIC
# MAGIC If you have extra time in your lab, feel free to make edits to the chain logic and config file to affect the behavior and performance of your RAG application!
