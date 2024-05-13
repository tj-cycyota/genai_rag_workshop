# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://companieslogo.com/img/orig/databricks_BIG-3be0f84a.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB 3. Creating and Deploying a RAG Chatbot
# MAGIC **Databricks GenAI Workshop 2024**
# MAGIC
# MAGIC Now that our data is ready, we can use it alongside a cutting-edge Large Language Model to answer user's questions. 
# MAGIC
# MAGIC In this notebook, we will:
# MAGIC
# MAGIC 1. Import simple LangChain components for:
# MAGIC     * [DatabricksVectorSearch](https://python.langchain.com/v0.1/docs/integrations/vectorstores/databricks_vector_search/): used to retrieve documents from our Vector Search Index from Step2
# MAGIC     * [DatabricksLLM](https://python.langchain.com/v0.1/docs/integrations/llms/databricks/): our actual large language model that receives an enriched prompt (with our retrieved documents) to answer a user's question
# MAGIC
# MAGIC 2. Build these components into a RAG Chain
# MAGIC
# MAGIC 3. Test the Chain in a Databricks Notebook
# MAGIC
# MAGIC 4. Log the Chain to Unity Catalog via MLflow
# MAGIC
# MAGIC 5. Load the Chain and perform Batch Inference
# MAGIC
# MAGIC 6. Deploy this Chain to a model serving endpoint for real-time requests

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12, 14.3.x-gpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %pip install -U --quiet mlflow langchain databricks-vectorsearch databricks-sdk mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

# Imports
import os
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.chat_models import ChatDatabricks

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Confirm variables; if these do not match what you created in Step 2, manually set them here or change the init file
print(f"Vector Search Endpoint: {vs_endpoint}")
print(f"Vector Search Index:    {full_index_location}")

# Setup host and tokens, which are used to interact with the various Databricks services
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_HOST"] = host
print(f"Current workspace: {host}")

# Setup temp session token. This is only for demo purposes; production applications should use a Service Principal. If you get token invalid errors, re-run this cell
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setting up Chatbot Chain components

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Search Retriever
# MAGIC
# MAGIC For RAG applications, it is common to create a `get_retriever()` function that is used to retrieve documents from a vector search database. 
# MAGIC
# MAGIC Notice how we do **not** need to specify the embedding model to use, as our Vector Search index was created with one. This is the simplicity of Databricks-managed embeddings in Vector Search - the client application (e.g. our LangChain) does not need to perform embeddings!

# COMMAND ----------

def get_retriever(persist_dir: str = None):
    # Setup Vector Search Client using our environment variables
    vsc = VectorSearchClient(
      workspace_url =         os.environ["DATABRICKS_HOST"], 
      personal_access_token = os.environ["DATABRICKS_TOKEN"],
      disable_notice=True
    )

    # Create pointer to our index
    vs_index = vsc.get_index(
        endpoint_name = vs_endpoint,
        index_name    = full_index_location
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, 
        text_column = "chunked_text"
    )
    return vectorstore.as_retriever()

# COMMAND ----------

# Test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("What is the expected accuracy of the thermocouples in the sensor?")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### DBRX LLM
# MAGIC We will use DBRX, a cutting edge open-source large language model. You are welcome to test other Foundation Models that Databricks hosts, including Llama 3. See docs: [Supported models for pay-per-token
# MAGIC ](https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html#dbrx-instruct)

# COMMAND ----------

chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)

print(f"Test chat results: \n {chat_model.predict('What is Apache Spark')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build RAG Chain
# MAGIC
# MAGIC In essence, a "chain" is a composition of multiple LLM-oriented classes that complete complex tasks. In our case, we are building a RAG chain: given a user question, retrieve similar documents to that question then pass an enriched prompt to a large language model to answer the question given the context. 
# MAGIC
# MAGIC There are many ways to build a RAG chain, and we'll use the simplest version from LangChain: [RetrievalQA](https://js.langchain.com/docs/modules/chains/popular/vector_db_qa_legacy). 
# MAGIC
# MAGIC This pattern is very extensable to more complex chains (such as Agents), which will all work on Databricks using the core components we have already learned! But for now, we will keep it simple. 

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

# Define a Prompt Template.
TEMPLATE = """You are an assistant for service technicians. You are answering questions related to technical devices and industrial machinery based on service manuals. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

# COMMAND ----------

# Define our Chain
chain = RetrievalQA.from_chain_type(
    llm =               chat_model, #dbrx foundation model
    chain_type =        "stuff", #stuff = RAG
    retriever =         get_retriever(), #Using our vector search retriever
    chain_type_kwargs = {"prompt": prompt} #prompt is passed as a Key-word argument 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test our Chain in the notebook
# MAGIC We can run a quick test before deploying our chatbot to a real-time API endpoint. 

# COMMAND ----------

import langchain
langchain.debug = False #First, try our prompt with no debug thread
question = {"query": "What is the expected accuracy of the thermocouples in the sensor?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC Great! Our model is producing the correct answer using the information in our PDF document!
# MAGIC
# MAGIC Now, let's see what is happening behind the scenes. We will enable `debug` in LangChain to see how the chain is executed, and what information is actually sent to the LLM:

# COMMAND ----------

langchain.debug = True
answer = chain.run(question) # Same question

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the above, and see how the original user's question is enriched with retrieved information before being sent to the LLM. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log the Chain to Unity Catalog via MLflow
# MAGIC
# MAGIC Now we have a working RAG Chain. But to use it to perform batch inference or as a real-time chatbot, we need to move beyond testing in a notebook. 
# MAGIC
# MAGIC To deploy our model to a scalable service to support external API calls, we will log the Model to MLflow. To learn more about LLM's and MLflow, see the [MLflow Documentation for LLMs](https://mlflow.org/docs/latest/llms/index.html), as well as the [MLflow LangChain flavor](https://mlflow.org/docs/latest/llms/langchain/index.html).

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain

# Set the MLflow registry to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog_name}.{schema_name}.product_manual_chatbot"
print(f"Logging model to UC location: {model_name}")

with mlflow.start_run(run_name="product_manual_chatbot") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch"
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load the Chain and perform Batch Inference
# MAGIC Navigate to the UC location printed out above, and confirm that you see a listing in the `Models` tab. This is the MLflow-compatible version of our model that we can load back up and serve. 
# MAGIC
# MAGIC To test that this works, and also to demonstrate how you can use this in batch inference, run the cell below. 

# COMMAND ----------

import pandas as pd
langchain.debug = False

# Load the first version of the model from UC
model_version_uri = "models:/"+model_name +"/1"
print(f"Loading model from UC: {model_version_uri}")

loaded_chain = mlflow.pyfunc.load_model(model_version_uri)

# Create simple dataset to demonstrate batch predictions
df = pd.DataFrame({"query": [
  "What is the expected accuracy of the thermocouples in the sensor?",
  "What are the component parts of Model 1400B?"
  ]})

# Call predict() method to get a new prediction
df["response"] = loaded_chain.predict(df)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Deploy this Chain to a model serving endpoint
# MAGIC
# MAGIC Databricks Model Serving allows us to deploy custom applications to a serverless endpoint that we can call as a REST API. Read more about it here: [Model serving with Databricks](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
# MAGIC
# MAGIC Before we deploy this model to model serving, we need to create a databricks secret so the Model Serving Endpoint can call back to the Vector Search Index we deployed. We'll use [Databricks Secrets](https://docs.databricks.com/en/security/secrets/secrets.html) for this process to avoid showing plain-text tokens.

# COMMAND ----------

# Change the Secret Scope name if needed
scope_name = current_user_safe+"_scope"
workspaceUrl = os.environ["DATABRICKS_HOST"]
databricks_token = os.environ["DATABRICKS_TOKEN"]

# See init notebook for supporting code
create_secrets(scope_name, workspaceUrl, databricks_token)

# We need 4 curly brackets due to use of Python f-strings
secret_host = f"{{{{secrets/{scope_name}/databricks_host}}}}"
secret_token = f"{{{{secrets/{scope_name}/databricks_token}}}}"
print(f"String env vars: {secret_host}, {secret_token}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy via UI
# MAGIC
# MAGIC To deploy our RAG Chain via UI:
# MAGIC * On the left-nav, click `Serving` then `Create serving endpoint`
# MAGIC * Name: choose a model serving name
# MAGIC * Entity: navigate to the UC model you logged (printed out above), then click Confirm
# MAGIC * Select Compute scale-out as `Small` and enable `Scale to zero`
# MAGIC * (Optional) If you want to see every request to the endpoint, enable `Inference tables`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy via Python SDK
# MAGIC
# MAGIC Before we deploy this model to model serving, we need to create a databricks secret so the Model Serving Endpoint can call back to the Vector Search Index we deployed. We'll use [Databricks Secrets](https://docs.databricks.com/en/security/secrets/secrets.html) for this process to avoid showing plain-text tokens.

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

serving_endpoint_name = f"{current_user_safe}_product_manual_chatbot"[:63]
# See init for function details
latest_model_version = get_latest_model_version(model_name)
print(f"Latest model version of {model_name}: {latest_model_version}")

# COMMAND ----------

w = WorkspaceClient()

endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_HOST": secret_host,
                "DATABRICKS_TOKEN": secret_token
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC The code above will eventually display a URL where you can visit your model serving endpoint. 
# MAGIC
# MAGIC In the meantime, you can click on the left nav to `Serving` to see it being provisioned. 
# MAGIC
# MAGIC When the endpoint is ready, click to it then `Query Endpoint` with `Sample Request` - feel free to make sure this matches the results you're seeing here in the notebook. 
# MAGIC
# MAGIC The important part is that external clients (e.g your custom chatbot front-end) can make REST API requests to this to serve chat applications. 
# MAGIC
# MAGIC Congrats - you've built a RAG chatbot from scratch!
