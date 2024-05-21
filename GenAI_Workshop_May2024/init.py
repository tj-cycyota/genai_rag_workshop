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
