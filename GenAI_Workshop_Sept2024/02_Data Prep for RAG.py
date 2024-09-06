# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://companieslogo.com/img/orig/databricks_BIG-3be0f84a.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB 2. Data Prep for Retrieval Augmented Generation
# MAGIC **Databricks GenAI Workshop**
# MAGIC
# MAGIC In this notebook we will prepare a PDF document for Vector Search, which will allow us to retrieve relevant documents for a RAG application.
# MAGIC
# MAGIC We'll apply this technique on a mostly text-based PDF related to technical product manuals to demonstrate the end-to-end processing steps:
# MAGIC 1. **Load** a document to a [UC Volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html). Normally, you would perform a batch or real-time process to move relevant documents (PDFs, HTMLs, XMLs, etc.) from an internal storage location into a Volume, but we'll do a simple UI-based workflow as well as programmatically retrieve public documents. 
# MAGIC
# MAGIC 2. **Extract** information from the PDF. We will use [pypdf](https://pypdf.readthedocs.io/en/stable/index.html) for this simple text-based example, but there are many other open-source and proprietary document-to-text options. 
# MAGIC
# MAGIC 3. **Split** the documents into usable "chunks". You will test multiple different chunking options. We will use a [LangChain Splitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/character_text_splitter/) in this simple example.
# MAGIC
# MAGIC 4. **Write** chunked data to a Delta table. We will do a one-time load of PDFs, but with [AutoLoader](https://docs.databricks.com/en/ingestion/auto-loader/unity-catalog.html), you can automate this process to always watch for newly-arriving files. 
# MAGIC
# MAGIC 5. **Synchronize** your "offline" Delta table with an "online" [Vector Search Index](https://docs.databricks.com/en/generative-ai/vector-search.html) that we will use for retrieval for a RAG application.
# MAGIC
# MAGIC 6. **Similarity Search** for documents based on a plain-English question to validate our implementation works!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s): **`15.4.x-cpu-ml-scala2.12, 15.4.x-gpu-ml-scala2.12`**

# COMMAND ----------

# MAGIC %md
# MAGIC First, run this pip install and initialization script to set a few parameters:

# COMMAND ----------

# MAGIC %pip install -U --quiet pypdf==4.1.0 databricks-vectorsearch transformers langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

import io
import re
from typing import List
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pypdf import PdfReader
from transformers import AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Use optimizations if available
dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
if dbr_majorversion >= 14:
  print("Setting arrow to enabled")
  spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load example PDF to a UC Volume
# MAGIC
# MAGIC This step can be done from the UI. 
# MAGIC 1. In a new browser tab, navigate to: 
# MAGIC * 
# MAGIC
# MAGIC 2. Download this PDF to your local laptop
# MAGIC
# MAGIC 3. On the left nav, navigate to: Catalog > Main > {your user-specific catalog} > Volumes > Create Volume > Upload to this Volume
# MAGIC
# MAGIC Alternatively, uncomment the cell below to run this step automatically:

# COMMAND ----------

volume_location = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/"
print(f"Documents will be put in UC at: {volume_location}")

# COMMAND ----------

import requests
import re

# Set the URL of the PDF file
pdf_url = "https://h10032.www1.hp.com/ctg/Manual/c05048181.pdf"
#"https://h10032.www1.hp.com/ctg/Manual/c01697043.pdf"

manual_filename = "manual.pdf"
path = volume_location+manual_filename

# Send a GET request to the URL
response = requests.get(pdf_url, stream=True, verify=False)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Open a local file in binary write mode
    with open(path, "wb") as f:
        # Iterate over the response content and write it to the file
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print(f"Download completed successfully, file saved to {path}")
else:
    print("Failed to download the PDF file. Check the URL or try again later.")

# COMMAND ----------

# MAGIC %md
# MAGIC Before proceeding: When you run the cell below, you should see (at least) 1 PDF file in your volume. If not, please follow the instructions above to place the PDF file in the UC Volume.

# COMMAND ----------

display(dbutils.fs.ls(volume_location))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Extract text from a PDF document
# MAGIC
# MAGIC We will start by reading in the entire Volume directory using a few settings. In a production setting, we would use [AutoLoader](https://docs.databricks.com/en/ingestion/auto-loader/unity-catalog.html) to automate the ingestion of new documents as they arrive.

# COMMAND ----------

print(f"Reading data from UC Volume: {volume_location}")

raw_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.pdf")
    .load(volume_location)
)

display(raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Observe that in our `raw_df` Spark DataFrame above, we have 1 row (for our 1 document), where the contents of that file are in the `content` column. You may have seen other `spark.read` formats, but it is common to use the `binaryFile` format when working with documents (PDFs, DOCX, etc.) then convert the binary contents later. 
# MAGIC
# MAGIC We can now parse out the binary contents! The below code uses a [Spark UDF](https://docs.databricks.com/en/udf/index.html) to extract information. While this code could be written as a traditional Python function, adding the `@F.udf()` decorator allows it to be parallelized across many workers if we have dozens/hundreds/millions of documents to process. 

# COMMAND ----------

@F.udf(
    returnType=StructType(
        [
            StructField("number_pages", IntegerType(), nullable=True),
            StructField("text", StringType(), nullable=True),
            StructField("status", StringType(), nullable=False),
        ]
    ),
    # useArrow=True, # set globally
)
def parse_pdf(pdf_raw_bytes):
    try:
        pdf = io.BytesIO(pdf_raw_bytes)
        reader = PdfReader(pdf)
        output_text = ""
        for _, page_content in enumerate(reader.pages):
            #PyPDF docs: https://pypdf.readthedocs.io/en/stable/user/extract-text.html
            output_text += page_content.extract_text( 
                extraction_mode="layout", 
                layout_mode_space_vertically=False,
                ) + "\n\n"

        return {
            "number_pages": len(reader.pages),
            "text": output_text,
            "status": "SUCCESS",
        }
    except Exception as e:
        return {"number_pages": None, "text": None, "status": f"ERROR: {e}"}

# COMMAND ----------

# Apply UDF to the binary "content" column
parsed_df = (
  raw_df.withColumn("parsed_output", parse_pdf("content"))
        .drop("content") # For brevity
        .drop("length")  # For brevity
)

display(parsed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Split the Document into Chunks
# MAGIC
# MAGIC If you expand the object in the `parsed_output` column from the last step, you will see we have a massive blob of raw text. 
# MAGIC
# MAGIC This is a relatively short document, so its not too long, but imagine if it were dozens or hundreds of pages long: it would be effectively unusable by an LLM, as every LLM has a maximum input length, and it is challenging for large language models to provide satisfactory responses when the prompt is too long. 
# MAGIC
# MAGIC So what should we do? **Chunk our documents**! This means splitting large blobs of text into shorter sections, which are usable by an LLM for question-answering. This approach is applicable to all kinds of documents (not just text-based ones): you can chunk large reference tables, mechanical diagrams, or really any kind of information you would find in a reference document. For our example, we'll stick with a straightforward approach and just break our big blob of text into smaller chunks, with some overlap between them. 
# MAGIC
# MAGIC In the cell below, try changing the first two variables to see how they affect the output dataframe. What settings make chunks that are too small? Too big? That are too "redundant" between them? There's no "right" answer here, so feel free to pick settings that make sense to you!

# COMMAND ----------

# Set chunking params
chunk_size_tokens = 2000
chunk_overlap_tokens = 200

# Instantiate tokenizer
## Read more here: https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#autotokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')

# Create UDF to recursively split text
## For other splitting approaches, see accompanying notebook
@F.udf(returnType=ArrayType(StringType())
          # useArrow=True, # set globally
          )
def split_char_recursive(content: str) -> List[str]:
    # Adding regex to remove ellipsis
    pattern = r'\.{3,}'
    cleaned_content = re.sub(pattern, '', content)
    # Use Hugging Face's CharacterTextSplitter
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, 
        separator = " ",
        chunk_size=chunk_size_tokens, 
        chunk_overlap=chunk_overlap_tokens
    )
    chunks = text_splitter.split_text(cleaned_content)
    return [doc for doc in chunks]

# Apply Chunking
chunked_df = (
  parsed_df.select(
    "*", 
    F.explode(split_char_recursive("parsed_output.text")).alias("chunked_text")
  )
  .drop(F.col("parsed_output"))
  .withColumn("chunk_id", F.md5(F.col("chunked_text")))
)

# Printouts to review results
num_chunks = chunked_df.count()
print(f"Number of chunks: {num_chunks}")

avg_chunk_words = chunked_df.withColumn("word_count", F.size(F.split(F.col("chunked_text"), " "))).select(F.avg(F.col("word_count"))).first()[0]
print(f"Average words per chunk: {avg_chunk_words}")

display(chunked_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment with Chunking Strategy
# MAGIC Now go back and change the number of tokens per chunk, and chunk overlap. 
# MAGIC
# MAGIC How does that affect:
# MAGIC * Number of rows ("chunks") in your resulting table? 
# MAGIC * Length in words of each chunk? Why is the number of words different than the `chunk_size_tokens` setting? Is there a relationship between a word and a token? 
# MAGIC * The "usefulness" of each chunk? Review the text of some of them. Are they too small to contain the useful information? Or too long that they're overly dense?
# MAGIC
# MAGIC If you have time during the lab, you can also check out other parsing/chunking libraries such as [unstructured.io](https://docs.unstructured.io/open-source/core-functionality/chunking) that allow you to extract information such as tables, charts, and images from the document. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write the final DataFrame to a Delta table
# MAGIC
# MAGIC We will save our final results to a table that we will then use as the basis for a vector search index to perform similarlity search. This step is pretty straightforward: we are persisting the results to a permanent location so we can re-use them later. 
# MAGIC
# MAGIC The code below will overwrite so you can run it more than once (but in production, you would append new chunks and/or merge existing ones). We need to enable [Change Data Feed](https://docs.databricks.com/en/delta/delta-change-data-feed.html) to allow Vector Search to monitor for changes tot his table

# COMMAND ----------

chunked_table_name = "chunked_product_manual"
full_table_location = f"{catalog_name}.{schema_name}.{chunked_table_name}"

print(f"Saving data to UC table: {full_table_location}")

(
  chunked_df.write
    .format("delta")
    .option("delta.enableChangeDataFeed", "true")
    .mode("overwrite")
    .saveAsTable(full_table_location)
)

# # We need to enable Change Data Feed on our Delta table to use it for Vector Search. If your table was already created, you can alter it:
# spark.sql(f"ALTER TABLE {full_table_location} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create a Vector Search Index of this table
# MAGIC
# MAGIC [Databricks Vector Search]() (generally available as of 5/8/2024) allows you to easily turn an "offline" Delta table into an "online" table capable of low-latency similarity search queries. You can read more [here](https://www.databricks.com/blog/production-quality-rag-applications-databricks)
# MAGIC
# MAGIC ![Vector Search diagram](https://docs.databricks.com/en/_images/calculate-embeddings.png "Vector Search diagram")
# MAGIC
# MAGIC [Vector Search Docs](https://docs.databricks.com/en/generative-ai/vector-search.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup via UI
# MAGIC You can perform this step from the UI: 
# MAGIC
# MAGIC 1. Navigate on the left side to Catalog > {your workshop catalog+schema} > Tables
# MAGIC
# MAGIC 2. Click on the table you created in the last step (e.g. `chunked_product_manuals`)
# MAGIC
# MAGIC 3. Once on the table screen: at the top-right, click Create > Vector Search Index
# MAGIC
# MAGIC 4. Fill in these details:
# MAGIC   * Enter index name: `product_manuals_index`
# MAGIC   * Primary key: `chunk_id`
# MAGIC   * Endpoint: `vs_endpoint_1` <-- replace `1` with a different number if errors occur
# MAGIC     * If you get an error creating, the index is full and you should create another one.
# MAGIC   * Embedding source: `Compute embeddings`
# MAGIC   * Embedding source column: `chunked_text`
# MAGIC   * Embedding model: `databricks-bge-large-en`
# MAGIC   * Sync mode: `Triggered`
# MAGIC
# MAGIC For more details, see documentation: [Create index using the UI
# MAGIC ](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-index-using-the-ui)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Vector Search Index programmatically.
# MAGIC `You can skip this step if you created the index in the UI`
# MAGIC
# MAGIC The below code completes two steps: 
# MAGIC 1. Sets up a Vector Search **Endpoint**. This is the **compute** that hosts your index, and an endpoint can host multiple indices
# MAGIC 2. Sets up a Vector Search **Index**. This is the **online replica** of your Delta table we will use in our RAG application
# MAGIC
# MAGIC Full documentation: [Create index using the Python SDK](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-index-using-the-python-sdk)
# MAGIC
# MAGIC **NOTE**: The cell below should run in 5-10 minutes, and will show as completed when the Endpoint is ready.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# Variable from init notebook
print(f"Vector Endpoint name: {vs_endpoint}. If issues, replace variable `vs_endpoint` with `fallback_vs_endpoint` in create_endpoint command")

# First, create the Endpoint
print(f"Attempting to find Endpoint named {vs_endpoint}; creating if new. This may take a few minutes...")
if vs_endpoint not in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]:
    vsc.create_endpoint(
        name=vs_endpoint, 
        endpoint_type="STANDARD"
        )

wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint)
print(f"Endpoint named {vs_endpoint} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE**: If you get an error above indicating that the workspace has too many endpoints, you should re-use an existing one that has open capacity (e.g. where # of indexes is less than `20`):
# MAGIC
# MAGIC * In the left-nav, go to `Compute` > `Vector Search`
# MAGIC * Find an Endpoint with open capacity. Copy the name of that Endpoint.
# MAGIC * In the cell above, replace the variable `vs_endpoint` with what you copied and re-run the cell.

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE**: The cell below should run in 5-10 minutes, and will show as completed when the Index is ready.

# COMMAND ----------

# One the endpoint is ready, lets create the Index
index_name = "product_index_"+current_user_safe
full_index_location = f"{catalog_name}.{schema_name}.{index_name}"

# Check first to see if index already exists
if not index_exists(vsc, vs_endpoint, full_index_location):
  print(f"Creating index {full_index_location} on endpoint {vs_endpoint}...")
  vsc.create_delta_sync_index(
    endpoint_name =                 vs_endpoint, # The endpoint where you want to host the index
    index_name =                    full_index_location, # Where in UC you want the index to be created
    source_table_name =             full_table_location, #The UC location of the offline source table
    pipeline_type =                 "TRIGGERED", # Set so we can manually refresh the index
    primary_key =                   "chunk_id", # The primary key of each chunk
    embedding_source_column =       "chunked_text", # The column containing chunked text
    embedding_model_endpoint_name = "databricks-bge-large-en" # The embedding model we want to use
  )
  # Creating the index will take a few moments. Navigate to the Catalog UI to take a look!
  wait_for_index_to_be_ready(vsc, vs_endpoint, full_index_location)
else:
  # If the index already exists, let's force a refresh using the .sync() method
  wait_for_index_to_be_ready(vsc, vs_endpoint, full_index_location)
  vsc.get_index(vs_endpoint, full_index_location).sync()

print(f"index {full_index_location} on table {full_table_location} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test Vector Search Index 
# MAGIC
# MAGIC Our index provides us the ability to perform **similarity search**: given a string, it will retrieve the documents in our index that most closely match that string. Let's give it a try:

# COMMAND ----------

# If you have ingested a different PDF document, change this question to something the document mentions
question = "Do I need to install any drivers on my computer to use the webscan feature?"
# "How do I scan a document?"

results = vsc.get_index(vs_endpoint, full_index_location).similarity_search(
  query_text=question,
  columns=["path", "chunked_text"],
  num_results=2)

docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the results, as well as the original document. Does the top-ranked result contain the answer to this question? Would you, as an expert technician, be able to answer the question given this extra information? Test num_results and see if multiple chunks contain this answer.
# MAGIC
# MAGIC Now that we've setup our data to be used in **retrieval augmented generation**, we can move on to build our question-answering chatbot in the next lab!
# MAGIC
# MAGIC **NOTE**: Make sure the steps above completed successfully, as they are needed in the next part of the lab!
# MAGIC
# MAGIC **Have extra time?**
# MAGIC * Review the `Chunking_Strategies` notebook in the `ExtraCredit` folder
# MAGIC * Explore and test other chunking libraries such as unstructured.io
