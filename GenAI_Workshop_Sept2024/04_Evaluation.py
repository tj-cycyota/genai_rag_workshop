# Databricks notebook source
# MAGIC %pip install --quiet -U databricks-agents mlflow mlflow-skinny databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

import pandas as pd
import mlflow
#w = WorkspaceClient()

#active_deployments = agents.list_deployments()
#active_deployment = next(
#    (item for item in active_deployments if item.model_name == rag_model_name), None
#)

#endpoint = w.serving_endpoints.get(active_deployment.endpoint_name)

#try:
#    endpoint_config = endpoint.config.auto_capture_config
#except AttributeError as e:
#    endpoint_config = endpoint.pending_config.auto_capture_config

inference_table_name = 'agent_framework_app_payload'
inference_table_catalog = 'main'
inference_table_schema = 'genai_workshop_wafa_louhichi'

# Cleanly formatted tables
assessment_table = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_assessment_logs`"
request_table = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_request_logs`"

# Note: you might have to wait a bit for the tables to be ready
print(f"Request logs: {request_table}")
requests_df = spark.table(request_table)
print(f"Assessment logs: {assessment_table}")
#Temporary helper to extract the table - see _resources/00-init-advanced 
assessment_df = deduplicate_assessments_table(assessment_table)

# COMMAND ----------

requests_with_feedback_df = requests_df.join(assessment_df, requests_df.databricks_request_id == assessment_df.request_id, "left")
display(requests_with_feedback_df.select("request_raw", "trace", "source", "text_assessment", "retrieval_assessments"))

# COMMAND ----------


requests_with_feedback_df.createOrReplaceTempView('latest_assessments')
eval_dataset = spark.sql(f"""
-- Thumbs up.  Use the model's generated response as the expected_response
select
  a.request_id,
  r.request,
  r.response as expected_response,
  'thumbs_up' as type,
  a.source.id as user_id
from
  latest_assessments as a
  join {request_table} as r on a.request_id = r.databricks_request_id
where
  a.text_assessment.ratings ["answer_correct"].value == "positive"
union all
  --Thumbs down.  If edited, use that as the expected_response.
select
  a.request_id,
  r.request,
  IF(
    a.text_assessment.suggested_output != "",
    a.text_assessment.suggested_output,
    NULL
  ) as expected_response,
  'thumbs_down' as type,
  a.source.id as user_id
from
  latest_assessments as a
  join {request_table} as r on a.request_id = r.databricks_request_id
where
  a.text_assessment.ratings ["answer_correct"].value = "negative"
union all
  -- No feedback.  Include the request, but no expected_response
select
  a.request_id,
  r.request,
  IF(
    a.text_assessment.suggested_output != "",
    a.text_assessment.suggested_output,
    NULL
  ) as expected_response,
  'no_feedback_provided' as type,
  a.source.id as user_id
from
  latest_assessments as a
  join {request_table} as r on a.request_id = r.databricks_request_id
where
  a.text_assessment.ratings ["answer_correct"].value != "negative"
  and a.text_assessment.ratings ["answer_correct"].value != "positive"
  """)
display(eval_dataset)

# COMMAND ----------

model = get_latest_model(rag_model_name)
pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{model.run_id}/chain")

# COMMAND ----------

# MAGIC %pip install -r $pip_requirements

# COMMAND ----------


with mlflow.start_run(run_name="eval_dataset"):
    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        data=eval_dataset,
        model=f'runs:/{model.run_id}/chain',
        model_type="databricks-agent",
    )

# COMMAND ----------

eval_results.tables['eval_results']

# COMMAND ----------

with mlflow.start_run(run_name="chatbot_rag_with_professionalism_metrics") as run:
    eval_results = mlflow.evaluate(data = eval_results.toPandas(),
                                   model_type="question-answering",           
                                   predictions="response",                  
                                   targets = "expected_response",
                                   extra_metrics=[answer_correctness_metrics, professionalism]) # Added professionalism metric
    
eval_results.metrics
