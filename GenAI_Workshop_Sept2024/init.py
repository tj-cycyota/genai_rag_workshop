# Databricks notebook source
# MAGIC %md
# MAGIC # Init notebook
# MAGIC This notebook initializes variables and UC locations for our RAG application. While it is recommended to implement config-driven applications, much of the code and helper functions here are for learning purposes to allow simple learning setups. 

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
# display(spark.sql("SHOW CURRENT SCHEMA"))

# Setup Volume for raw PDFs
volume_name = "product_manuals"
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{volume_name}")

print(f"Place PDFs for RAG in: {catalog_name}.{schema_name}.{volume_name}")

# COMMAND ----------

# Function used to randomly assign each user a VS Endpoint
def get_fixed_integer(string_input):
    # Calculate the sum of ASCII values of the characters in the input string
    ascii_sum = sum(ord(char) for char in string_input)
    
    # Map the sum to a fixed integer between 1 and 9
    fixed_integer = (ascii_sum % 9) + 1
    
    return fixed_integer

# COMMAND ----------

vector_search_endpoint_prefix = "vs_endpoint_"

# Picking Vector Search Endpoint, a shared resource b/w participants
vs_endpoint = vector_search_endpoint_prefix+str(get_fixed_integer(schema_name))
fallback_vs_endpoint = vector_search_endpoint_prefix+schema_name
print(f"Vector Endpoint name: {vs_endpoint}. If issues, replace variable `vs_endpoint` with `fallback_vs_endpoint` in Labs 2+3.")

index_name = "product_index_"+current_user_safe
full_index_location = f"{catalog_name}.{schema_name}.{index_name}"
print(f"Vector Index name: {full_index_location}")

import time
def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

## UC Model name where the POC chain is logged
rag_model_name = f"{catalog_name}.{schema_name}.agent_framework_app"
EVALUATION_SET_FQN = f"{rag_model_name}_evaluation_set"

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
    indexes = vsc.list_indexes(endpoint_name).get("vector_indexes", list())
    if any(index_full_name == index.get("name") for index in indexes):
      return True
    #Temp fix when index is not available in the list
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get('status').get('ready')
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False
    
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(20)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

def create_secrets(scope_name, workspaceUrl, databricks_token):
  import requests, json
  headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}

  payload_host = {
          "scope": scope_name,
          "key": "databricks_host",
          "string_value": workspaceUrl
      }
  payload_token = {
            "scope": scope_name,
            "key": "databricks_token",
            "string_value": databricks_token
      }

  # Check if secret scope already exists, otherwise create it
  try:
    response = requests.get(f'{workspaceUrl}/api/2.0/secrets/list?scope={scope_name}', headers=headers)
    if response.status_code == 200:
      # Reset host and token at scope
      response_host = requests.post(f'{workspaceUrl}/api/2.0/secrets/put', headers=headers, json=payload_host)
      response_token = requests.post(f'{workspaceUrl}/api/2.0/secrets/put', headers=headers, json=payload_token)
      print(f"Patched host and token. Secrets at scope: {scope_name}")
      print(json.dumps(response.json(),indent=4))
    if response.status_code == 404:
      print(f"{scope_name} scope not created yet; creating....")
      payload_scope = {
          "scope": scope_name,
          "scope_backend_type": "DATABRICKS"
        }
      response = requests.post(f'{workspaceUrl}/api/2.0/secrets/scopes/create', headers=headers, json=payload_scope)
      response = requests.get(f'{workspaceUrl}/api/2.0/secrets/list?scope={scope_name}', headers=headers)
      print(f"Secrets at scope: {scope_name}")
      print(json.dumps(response.json(),indent=4))

      # Put host and token at scope
      response_host = requests.post(f'{workspaceUrl}/api/2.0/secrets/put', headers=headers, json=payload_host)
      response_token = requests.post(f'{workspaceUrl}/api/2.0/secrets/put', headers=headers, json=payload_token)
  except requests.exceptions.RequestException as e:
      print("An error occurred during the API call:", e)
    
  # Check if host and token can be retrieved 
  try:
    print("Attempting to get temporary tokens to workspace, should show redacted...")
    print(dbutils.secrets.get(scope_name,'databricks_host'))
    print(dbutils.secrets.get(scope_name,'databricks_token'))
  except requests.exceptions.RequestException as e:
      print("An error occurred during the dbutils call:", e)
    

  # print("Will show redacted if secrets properly created")
  # print(dbutils.secrets.get(scope_name,'databricks_host'))
  # print(dbutils.secrets.get(scope_name,'databricks_token'))

# COMMAND ----------

from mlflow import MlflowClient
def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

from typing import Dict, Any

def _flatten_nested_params(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, str]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_nested_params(v, new_key, sep=sep))
        else:
          items[new_key] = v
    return items

def tag_delta_table(table_fqn, config):
    flat_config = _flatten_nested_params(config)
    sqls = []
    for key, item in flat_config.items():
        
        sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("{key.replace("/", "__")}" = "{item}")
        """)
    sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("table_source" = "rag_poc_pdf")
        """)
    for sql in sqls:
        # print(sql)
        spark.sql(sql)

# COMMAND ----------

# Chain configuration
# We suggest using these default settings
rag_chain_config = {
    "databricks_resources": {
        # Only required if using Databricks vector search
        "vector_search_endpoint_name": vs_endpoint,
        # Databricks Model Serving endpoint name
        # This is the generator LLM where your LLM queries are sent.
        "llm_endpoint_name": "databricks-dbrx-instruct",
    },
    "retriever_config": {
        # Vector Search index that is created by the data pipeline
        "vector_search_index": full_index_location,
        "schema": {
            # The column name in the retriever's response referred to the unique key
            # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
            "primary_key": "chunk_id",
            # The column name in the retriever's response that contains the returned chunk.
            "chunk_text": "chunked_text",
            # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
            "document_uri": "path",
        },
        # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question
        "chunk_template": "Passage: {chunk_text}\n",
        # The column name in the retriever's response that refers to the original document.
        "parameters": {
            # Number of search results that the retriever returns
            "k": 5,
            # Type of search to run
            # Semantic search: `ann`
            # Hybrid search (keyword + sementic search): `hybrid`
            "query_type": "ann",
        },
        # Tag for the data pipeline, allowing you to easily compare the POC results vs. future data pipeline configurations you try.
        "data_pipeline_tag": "poc",
    },
    "llm_config": {
        # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
        "llm_system_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.

Context: {context}""".strip(),
        # Parameters that control how the LLM responds.
        "llm_parameters": {"temperature": 0.01, "max_tokens": 1500},
    },
    "input_example": {
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
        ]
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config & save to YAML

# COMMAND ----------

import yaml, json
print(f"Using chain config: {json.dumps(rag_chain_config, indent=4)}")

with open('rag_chain_config.yaml', 'w') as f:
    yaml.dump(rag_chain_config, f)

# COMMAND ----------

# MAGIC %md
# MAGIC Helpers for Evaluation

# COMMAND ----------

def deduplicate_assessments_table(assessment_table):
    # De-dup response assessments
    assessments_request_deduplicated_df = spark.sql(f"""select * except(row_number)
                                        from ( select *, row_number() over (
                                                partition by request_id
                                                order by
                                                timestamp desc
                                            ) as row_number from {assessment_table} where text_assessment is not NULL
                                        ) where row_number = 1""")
    # De-dup the retrieval assessments
    assessments_retrieval_deduplicated_df = spark.sql(f"""select * except( retrieval_assessment, source, timestamp, text_assessment, schema_version),
        any_value(timestamp) as timestamp,
        any_value(source) as source,
        collect_list(retrieval_assessment) as retrieval_assessments
      from {assessment_table} where retrieval_assessment is not NULL group by request_id, source.id, step_id"""    )

    # Merge together
    assessments_request_deduplicated_df = assessments_request_deduplicated_df.drop("retrieval_assessment", "step_id")
    assessments_retrieval_deduplicated_df = assessments_retrieval_deduplicated_df.withColumnRenamed("request_id", "request_id2").withColumnRenamed("source", "source2").drop("step_id", "timestamp")

    merged_deduplicated_assessments_df = assessments_request_deduplicated_df.join(
        assessments_retrieval_deduplicated_df,
        (assessments_request_deduplicated_df.request_id == assessments_retrieval_deduplicated_df.request_id2) &
        (assessments_request_deduplicated_df.source.id == assessments_retrieval_deduplicated_df.source2.id),
        "full"
    ).select(
        [str(col) for col in assessments_request_deduplicated_df.columns] +
        [assessments_retrieval_deduplicated_df.retrieval_assessments]
    )

    return merged_deduplicated_assessments_df

# COMMAND ----------

# Helper function
def get_latest_model(model_name):
    from mlflow.tracking import MlflowClient
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = None
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if not latest_version or version_int > int(latest_version.version):
            latest_version = mv
    return latest_version
