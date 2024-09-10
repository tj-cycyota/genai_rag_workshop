# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://companieslogo.com/img/orig/databricks_BIG-3be0f84a.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB 4. Turn the Review App logs into an Evaluation Set
# MAGIC **Databricks GenAI Workshop**
# MAGIC
# MAGIC The Review application captures feedback from your expert reviewers, a crucial testing step before releasing a Compound AI application to production. This feedback is saved under 2 tables within your schema.
# MAGIC
# MAGIC **NOTE: feedback to the inference tables may take up to 2 hours to populate!**
# MAGIC
# MAGIC In this notebook, we will show you how to extract the logs from the Review App into an Evaluation Set.  It is important to review each row and ensure the data quality is high e.g., the question is logical and the response makes sense.
# MAGIC
# MAGIC 1. Requests with a 👍 :
# MAGIC     - `request`: As entered by the user
# MAGIC     - `expected_response`: If the user edited the response, that is used, otherwise, the model's generated response.
# MAGIC 2. Requests with a 👎 :
# MAGIC     - `request`: As entered by the user
# MAGIC     - `expected_response`: If the user edited the response, that is used, otherwise, null.
# MAGIC 3. Requests without any feedback
# MAGIC     - `request`: As entered by the user
# MAGIC
# MAGIC Across all types of requests, if the user 👍 a chunk from the `retrieved_context`, the `doc_uri` of that chunk is included in `expected_retrieved_context` for the question.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F03-advanced-app%2F03-Offline-Evaluation&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F03-advanced-app%2F03-Offline-Evaluation&version=1">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s): **`15.4.x-cpu-ml-scala2.12, 15.4.x-gpu-ml-scala2.12`**
# MAGIC
# MAGIC First, run this pip install and initialization script to set a few parameters:

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow mlflow-skinny databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Extracting the logs from Inference Tables
# MAGIC
# MAGIC Anytime a subject matter expert logs feedback via the Review App, that feedback is captured in the Inference Table (e.g. Delta table) deployed with your Model Serving Endpoint. This feedback is exactly what we need to evaluate if our RAG application is behaving as expected, as well as assess the impacts of any changes we make to the application logic. 
# MAGIC
# MAGIC **If you are running this code in a lab environment and deployed your application <2 hours ago, skip ahead to section 2**
# MAGIC
# MAGIC For now, this part requires a few SQL queries that we provide in this notebook to properly format the review app into training dataset. This will be simpler in the future.

# COMMAND ----------

import os
import mlflow
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import pandas_udf, col

inference_table_name = 'agent_framework_app_payload'

# Cleanly formatted tables
assessment_table = f"{catalog_name}.{schema_name}.`{inference_table_name}_assessment_logs`"
request_table = f"{catalog_name}.{schema_name}.`{inference_table_name}_request_logs`"

# Note: you might have to wait a bit for the tables to be ready
print(f"Request logs: {request_table}")
requests_df = spark.table(request_table)
print(f"Assessment logs: {assessment_table}")
#Temporary helper to extract the table - see _resources/00-init-advanced 
assessment_df = deduplicate_assessments_table(assessment_table)

requests_with_feedback_df = requests_df.join(assessment_df, requests_df.databricks_request_id == assessment_df.request_id, "left")
display(requests_with_feedback_df.select("request_raw", "trace", "source", "text_assessment", "retrieval_assessments"))

# COMMAND ----------

requests_with_feedback_df.createOrReplaceTempView("latest_assessments")
eval_dataset = spark.sql(
    f"""
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
  """
)

# COMMAND ----------

# MAGIC %md 
# MAGIC **Our eval dataset is now ready!**
# MAGIC
# MAGIC The review app makes it easy to build and create your evaluation dataset. 
# MAGIC
# MAGIC *Note: the eval app logs may take some time to be available to you. If the dataset is empty, wait a bit or skip ahead to use the offline dataset.*

# COMMAND ----------

display(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Offline Feedback Data
# MAGIC
# MAGIC To simplify the demo and make sure you don't have to craft your own eval dataset, we saved a ready-to-use eval dataset already pre-generated for you. We'll use this one for the demo instead.

# COMMAND ----------

filename = "Data/Agent_Evaluation_Dataset.csv"
data_path = "file:" + os.path.join(os.getcwd(), filename)
print(data_path)  # See file in Data folder

eval_dataset = spark.read.csv(data_path, header=True, inferSchema=True, lineSep="\n", multiLine=True, quote="\"", escape="\"")

display(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Mosaic Agent Evaluation 
# MAGIC
# MAGIC Using [MLflow Evaluate](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html) and [Mosaic Agent Evaluation](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html), we will quantitatively measure the quality of our application using the ground-truth dataset provided by the review app. When we talk about "evaluation driven development", this is what we mean - using a framework that allows us to incrementally improve performance of our compound AI system. 
# MAGIC
# MAGIC This is a key area where Databricks is differentiated from open source offerings: a highly curated, fine-tuned evaluation agent to assess system performance.
# MAGIC
# MAGIC Let's first make sure our evaluation environment matches the environment in which the model was created to make sure our results are reproducible.

# COMMAND ----------

model = get_latest_model(rag_model_name)
pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{model.run_id}/chain")

# COMMAND ----------

# MAGIC %pip install --quiet -r $pip_requirements

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

# MAGIC %md
# MAGIC You can open MLFlow and review the eval metrics, and also compare it to previous eval runs!
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/mlflow-eval.gif?raw=true" width="1200px"> 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. OSS MLflow Evaluation
# MAGIC
# MAGIC In addition to Mosaic Agent Framework, we can use open-source MLflow evaluate to see how our system is performing. 
# MAGIC
# MAGIC First, we need to re-label a few of the columns in our dataset.

# COMMAND ----------

from pyspark.sql.functions import expr
test_data = eval_dataset.select(
    expr("request_id").alias("id"),
    expr("request").alias("inputs"),
    expr("expected_response").alias("ground_truth")
)

display(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Run inference to get predictions and create the evaluation dataset

# COMMAND ----------

# Set the MLflow registry to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog_name}.{schema_name}.agent_framework_app"

model_version_to_evaluate = get_latest_model_version(model_name)
print(f"Latest model version of {model_name}: {model_version_to_evaluate}")

# Load Model to make predictions
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

# COMMAND ----------

# To switch things up, we'll switch to Pandas for inference
test_data_pd = test_data.toPandas()

res = []
for input_col in test_data_pd.inputs:
  chain_input = {
  "messages": [
    {
    "role": "user",
    "content": input_col,
    }
    ]}
  res.append(rag_model.invoke(chain_input))

# Populate our results
test_data_rag_answers = test_data_pd.copy()
test_data_rag_answers["rag_answer"] = res
display(test_data_rag_answers)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4.2 Define Metrics
# MAGIC
# MAGIC MLflow 2.8 provides out of the box GenAI metrics and enables us to make our own GenAI metrics:
# MAGIC  - Mlflow will automatically compute relevant task-related metrics. In our case, `model_type='question-answering'` will add the `toxicity` and `token_count` metrics.
# MAGIC  - Then, we can import out of the box metrics provided by MLflow 2.8. Let's benefit from our ground-truth labels by computing the `answer_correctness` metric. 
# MAGIC  - Finally, we can define customer metrics. Here, creativity is the only limit. In our demo, we will evaluate the `professionalism` of our Q&A chatbot.

# COMMAND ----------

from mlflow.metrics.genai.metric_definitions import answer_correctness
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# We will use a different model than our RAG application is using to evaluate its performance
# To learn more about Llama-3: https://www.databricks.com/blog/building-enterprise-genai-apps-meta-llama-3-databricks
endpoint_name = "databricks-meta-llama-3-70b-instruct"

# Because we have our labels (ground-truth answers) within the evaluation dataset, we can evaluate the answer correctness as part of our metric. 
answer_correctness_metrics = answer_correctness(model=f"endpoints:/{endpoint_name}")

# Optional: review the plain-text logic of how we are evaluating "correctness"
print(answer_correctness_metrics)

# COMMAND ----------

# Review help string for mlflow.evaluate - this can also be used on traditional ML problems!
help(mlflow.evaluate)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Adding in a custom metric
# MAGIC - LLM-as-a-judge is a powerful tool to evaluate the behavior of a RAG application at scale. We've seen how the model can evaluate a standard set of more qualitative metrics in a standardized and performant way. But what about custom metrics? 
# MAGIC - Let's take the idea of "professionalism", something we can all agree is important in a corporate question-answering scenario. Below, we will first 1) provide an example then 2) define the metric. 
# MAGIC

# COMMAND ----------

# Create high-scoring example
professionalism_example_high = EvaluationExample(
    input="Where can I learn more about warranties?",
    output=(
        "We care about our customers, and warrant our goods as being free of defective materials and faulty workmanship. Our standard product warranty applies unless agreed to otherwise in writing. Please refer to your order acknowledgement or consult your local sales office for specific warranty details."
    ),
    score=5,
    justification=(
        "The response is written in a clear and professional tone, choosing words that would be used in a corporate setting. The answer leaves no room for ambiguity. It does not use contractions, filler words, or exclamation points or slang."
    )
)

# Create low-scoring example
professionalism_example_low = EvaluationExample(
    input="Where can I learn more about warranties?",
    output=(
        "Yo, bro! If you wanna know more 'bout warranties, you gotta check out our sick website, man! It's got all the deets you need. Just hit up the URL and you'll find all the info you're lookin' for. No worries, dude!"
    ),
    score=1,
    justification=(
        "The response uses slang, colloquialisms, contractions, exclamation points, and otherwise non-professional words and phrases."
    )
)

# Define our custom metric
professionalism = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is tailored to the context and audience. It often involves avoiding overly casual language, slang, or colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    model=f"endpoints:/{endpoint_name}",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    examples=[professionalism_example_high, professionalism_example_low],
    greater_is_better=True
)

print(professionalism)


# COMMAND ----------

# Create new run with extra metrics
with mlflow.start_run(run_name="chatbot_rag_with_professionalism_metrics") as run:
    eval_results = mlflow.evaluate(data = test_data_rag_answers,
                                   model_type="question-answering",           
                                   predictions="rag_answer",                  
                                   targets = "ground_truth",
                                   extra_metrics=[answer_correctness_metrics, professionalism]) # Added professionalism metric
    
eval_results.metrics

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ###  4.4 Navigate to the MLflow Evaluation UI to see how our model performed on this Professionalism metric!
# MAGIC  Now that you have the hang of how to use `mlflow.evaluate()`, feel free to apply this concept to more questions and/or more custom metrics. 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Getting ready for Production
# MAGIC
# MAGIC After reviewing the model correctness and potentially comparing its behavior to your other previous version, we can flag our model as ready to be deployed.
# MAGIC
# MAGIC *Note: Evaluation can be automated and part of an MLOps process: once you deploy a new Chatbot version with a new prompt, run the evaluation job and benchmark your model behavior vs the previous version.*

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(name=model_name, alias="prod", version=model_version_to_evaluate)

# COMMAND ----------

# MAGIC %md
# MAGIC This lab notebook is a small example of an automated approach to LLM/GenAI Evaluation. For a deeper dive, we recommend The Databricks Generative AI Cookbook is a definitive how-to guide for building high-quality generative AI applications:
# MAGIC
# MAGIC **--> [Databricks Generative AI Cookbook](https://ai-cookbook.io/index.html) <--**
# MAGIC
# MAGIC As a continuation of the application you have already built, check out the section on [`RAG Quality Knobs`](https://ai-cookbook.io/nbs/3-deep-dive.html):
# MAGIC
# MAGIC ![data pipeline diagram](https://ai-cookbook.io/_images/15_img.png "data pipeline diagram")
# MAGIC
# MAGIC ![rag chain diagram](https://ai-cookbook.io/_images/16_img.png "rag chain diagram")
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonus: Custom visualizations
# MAGIC
# MAGIC You can plot the evaluation metrics directly from the run, or pulling the data from MLFlow. 
# MAGIC
# MAGIC (Feel free to also use the built-in notebook visualizations by clicking the `+` sign after running a `display()` command)
# MAGIC

# COMMAND ----------

df_genai_metrics = eval_results.tables["eval_results_table"]
display(df_genai_metrics)

# COMMAND ----------

import plotly.express as px
px.histogram(df_genai_metrics, x="token_count", labels={"token_count": "Token Count"}, title="Distribution of Token Counts in Model Responses")


# COMMAND ----------

# Counting the occurrences of each answer correctness score
px.bar(df_genai_metrics['answer_correctness/v1/score'].value_counts(), title='Answer Correctness Score Distribution')


# COMMAND ----------

df_genai_metrics['toxicity'] = df_genai_metrics['toxicity/v1/score'] * 100
fig = px.scatter(df_genai_metrics, x='toxicity', y='answer_correctness/v1/score', title='Toxicity vs Correctness', size=[10]*len(df_genai_metrics))
fig.update_xaxes(tickformat=".2f")
