# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://companieslogo.com/img/orig/databricks_BIG-3be0f84a.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluating a RAG Application
# MAGIC **Databricks GenAI Workshop 2024**
# MAGIC
# MAGIC We've created a RAG Application, but how do you know if its not only "good", but more importantly, **ready to deploy to production?**
# MAGIC
# MAGIC In this notebook, we will:
# MAGIC
# MAGIC 1. **Create Sample Evaluation data** with "ground-truth" data from a human evaluator
# MAGIC
# MAGIC 2. **Apply our RAG chain** to this evaluation data
# MAGIC
# MAGIC 3. **Assess "Correctness" Using another Large Language Model**
# MAGIC
# MAGIC 4. **Visualize our Evaluation metrics in MLflow**
# MAGIC
# MAGIC 5. **Create a custom "Professionalism" Metric** and evaluate our model
# MAGIC
# MAGIC MLflow Evaluate is a powerful tool to help enterprises gain confidence in releasing Generative AI applications, such as RAG. To learn more, review the docs here: [MLflow LLM Evaluate](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)

# COMMAND ----------

# MAGIC %pip install -U --quiet mlflow databricks-vectorsearch langchain==0.2.0 langchain-community==0.2.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../init

# COMMAND ----------

# Imports
import os
import mlflow
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import pandas_udf, col

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
# MAGIC ## 1. Evaluation Data
# MAGIC
# MAGIC First, let's create a small sample dataset showcasing the "human approved" (a.k.a. ground-truth) answers to questions. These are answers you would have already collected, or collect alongside the development process of a RAG application to make sure a RAG Application is providing quality responses. Additionally, this toy example creates a few rows in a notebook; in a production setting, you should retrieve these ground-truth answers from a well-curated Delta table.

# COMMAND ----------

# Define dataframe schema
schema = StructType([
    StructField("id", StringType(), nullable=False),
    StructField("inputs", StringType(), nullable=False),
    StructField("ground_truth", StringType(), nullable=False)
])

# Create a few rows of sample ground-truth data
data = [
    ("1", "What is the expected accuracy of the thermocouples in the sensor?", 
     "Typically +/-1.7°C over the measurement range."),
    
    ("2", "What are the component parts of Model 1400B?", 
     "The component parts of Model 1400B include the Model 1418B transmitter, 1442B receiver, 1420A power supply, 1427A rotating antenna, and 1408 stationary antenna."),
    
    ("3", "Is the system affected by electrical noise or interference?", 
     "No the system is unaffected by the electrical noise of other equipment.")
]

test_data = spark.createDataFrame(data, schema)
display(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Make predictions with RAG app
# MAGIC
# MAGIC We will create a new column in our dataframe with the predicted answer from our RAG application

# COMMAND ----------

# Set the MLflow registry to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog_name}.{schema_name}.product_manual_chatbot"

model_version_to_evaluate = get_latest_model_version(model_name)
print(f"Latest model version of {model_name}: {model_version_to_evaluate}")

# Load Model to make predictions
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

# A pandas_udf helps apply this RAG model at scale across a cluster
@pandas_udf("string")
def predict_answer(questions):
    def answer_question(question):
        payload = {"query": question}
        return rag_model.invoke(payload)['result']
    return questions.apply(answer_question)
  
# Apply RAG app to our test data: rag_answer is what our RAG app returns
test_data_rag_answers = test_data.withColumn("rag_answer", predict_answer(col("inputs")))
display(test_data_rag_answers)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3. LLMs-as-a-judge: automated LLM evaluation with out of the box and custom GenAI metrics
# MAGIC
# MAGIC MLflow 2.8 provides out of the box GenAI metrics and enables us to make our own GenAI metrics:
# MAGIC - Mlflow will automatically compute relevant task-related metrics. In our case, `model_type='question-answering'` will add the `toxicity` and `token_count` metrics.
# MAGIC - Then, we can import out of the box metrics provided by MLflow 2.8. Let's benefit from our ground-truth labels by computing the `answer_correctness` metric. 
# MAGIC - Finally, we can define customer metrics. Here, creativity is the only limit. In our demo, we will evaluate the `professionalism` of our Q&A chatbot.
# MAGIC

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

from mlflow.deployments import set_deployments_target
set_deployments_target("databricks")

#This will automatically log all answers to MLflow evaluation UI
with mlflow.start_run(run_name="chatbot_rag_correctness") as run:
    eval_results = mlflow.evaluate(data = test_data_rag_answers.toPandas(),   # evaluation data,
                                   model_type="question-answering",           # toxicity and token_count will be evaluated   
                                   predictions="rag_answer",                  # prediction column_name from eval_df
                                   targets = "ground_truth",
                                   extra_metrics=[answer_correctness_metrics]) # add extra_metric for correctness
    
eval_results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Visualization of our GenAI metrics produced by our GPT4 judge
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-llm-as-a-judge-mlflow.png?raw=true" style="float: right; margin-left:10px" width="800px">
# MAGIC
# MAGIC You can open your MLFlow experiment runs from the Experiments menu on the right. Open the experiment, then navigate to the "Evaluation" tab. 
# MAGIC
# MAGIC From here, you can compare multiple model versions, and filter by correctness to spot where your model doesn't answer well. 
# MAGIC
# MAGIC Based on that and depending on the issue, you can either fine tune your prompt, your model fine tuning instruction with RLHF, or improve your documentation.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##5. Adding in a custom metric
# MAGIC
# MAGIC LLM-as-a-judge is a powerful tool to evaluate the behavior of a RAG application at scale. We've seen how the model can evaluate a standard set of more qualitative metrics in a standardized and performant way. But what about custom metrics? 
# MAGIC
# MAGIC Let's take the idea of "professionalism", something we can all agree is important in a corporate question-answering scenario. Below, we will first 1) provide an example then 2) define the metric. 

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
    eval_results = mlflow.evaluate(data = test_data_rag_answers.toPandas(),
                                   model_type="question-answering",           
                                   predictions="rag_answer",                  
                                   targets = "ground_truth",
                                   extra_metrics=[answer_correctness_metrics, professionalism]) # Added professionalism metric
    
eval_results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC Navigate to the MLflow Evaluation UI to see how our model performed on this Professionalism metric!
# MAGIC
# MAGIC Now that you have the hang of how to use `mlflow.evaluate()`, feel free to apply this concept to more questions and/or more custom metrics. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus: Custom visualizations
# MAGIC You can plot the evaluation metrics directly from the run, or pulling the data from MLFlow. 
# MAGIC
# MAGIC (Feel free to also use the built-in notebook visualizations by clicking the `+` sign after running a `display()` command)

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

# COMMAND ----------

# MAGIC %md
# MAGIC ##6. Getting ready for Production
# MAGIC
# MAGIC After reviewing the model correctness and potentially comparing its behavior to your other previous version, we can flag our model as ready to be deployed.
# MAGIC
# MAGIC *Note: Evaluation can be automated and part of a MLOps step: once you deploy a new Chatbot version with a new prompt, run the evaluation job and benchmark your model behavior vs the previous version.*

# COMMAND ----------

# client = MlflowClient()
# client.set_registered_model_alias(name=model_name, alias="prod", version=model_version_to_evaluate)
