# Databricks notebook source
# MAGIC %md
# MAGIC ## Figuring out the best chunk size for your application
# MAGIC
# MAGIC
# MAGIC
# MAGIC Here are some pointers to help you come up with an optimal chunk size if the common chunking approaches, like fixed chunking, don’t easily apply to your use case :
# MAGIC
# MAGIC - Preprocessing your Data - You need to first pre-process your data to ensure quality before determining the best chunk size for your application. For example, if your data has been retrieved from the web, you might need to remove HTML tags or specific elements that just add noise.
# MAGIC - Selecting a Range of Chunk Sizes - Once your data is preprocessed, the next step is to choose a range of potential chunk sizes to test. As mentioned previously, the choice should take into account the nature of the content (e.g., short messages or lengthy documents), the embedding model you’ll use, and its capabilities (e.g., token limits). The objective is to find a balance between preserving context and maintaining accuracy. Start by exploring a variety of chunk sizes, including smaller chunks (e.g., 128 or 256 tokens) for capturing more granular semantic information and larger chunks (e.g., 512 or 1024 tokens) for retaining more context.
# MAGIC - Evaluating the Performance of Each Chunk Size - In order to test various chunk sizes, you can either use multiple indices or a single index with multiple namespaces. With a representative dataset, create the embeddings for the chunk sizes you want to test and save them in your index (or indices). You can then run a series of queries for which you can evaluate quality, and compare the performance of the various chunk sizes. This is most likely to be an iterative process, where you test different chunk sizes against different queries until you can determine the best-performing chunk size for your content and expected queries.
# MAGIC
# MAGIC [Source](https://www.pinecone.io/learn/chunking-strategies/)
# MAGIC

# COMMAND ----------

# MAGIC %pip install nltk langchain 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import nltk
from nltk.tokenize import sent_tokenize

# Sample text from the blog about chunking strategies
sample_text = """
Databricks Vector Search is a vector database that is built into the Databricks Intelligence Platform and integrated with its governance and productivity tools. A vector database is a database that is optimized to store and retrieve embeddings. Embeddings are mathematical representations of the semantic content of data, typically text or image data. Embeddings are generated by a large language model and are a key component of many GenAI applications that depend on finding documents or images that are similar to each other. Examples are RAG systems, recommender systems, and image and video recognition.

With Vector Search, you create a vector search index from a Delta table. The index includes embedded data with metadata. You can then query the index using a REST API to identify the most similar vectors and return the associated documents. You can structure the index to automatically sync when the underlying Delta table is updated.

Databricks Vector Search uses the Hierarchical Navigable Small World (HNSW) algorithm for its approximate nearest neighbor searches and the L2 distance distance metric to measure embedding vector similarity. If you want to use cosine similarity you need to normalize your datapoint embeddings before feeding them into Vector Search. When the data points are normalized, the ranking produced by L2 distance is the same as the ranking produces by cosine similarity.
"""



# COMMAND ----------

# DBTITLE 1,Sentence splitting using Naive method
docs = sample_text.split(".")
for i, doc in enumerate(docs):
    print(f"Chunk {i+1}: {doc}")


# COMMAND ----------

# MAGIC %md
# MAGIC This is the most common and straightforward approach to chunking. We simply decide the number of tokens in our chunk and, optionally, whether there should be any overlap between them. It is computationally cost-effective

# COMMAND ----------

# DBTITLE 1,Fixed Size Chunking
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = " ",
    chunk_size = 256,
    chunk_overlap  = 20
)
docs = text_splitter.create_documents([sample_text])
for i, doc in enumerate(docs):
    print(f"Chunk {i+1}: {doc}")


# COMMAND ----------

# MAGIC %md
# MAGIC Recursive chunking divides the input text into smaller chunks in a hierarchical and iterative manner using a set of separators

# COMMAND ----------

# DBTITLE 1,Recursive Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 256,
    chunk_overlap  = 20
)

docs = text_splitter.create_documents([sample_text])

for i, doc in enumerate(docs):
    print(f"Chunk {i+1}: {doc}")

# COMMAND ----------

# MAGIC %md
# MAGIC Markdown and LaTeX are two examples of structured and formatted content you might run into. In these cases, you can use specialized chunking methods to preserve the original structure of the content during the chunking process. This is to get more semnatically coherent chunks as it is dividing based on structure and heirarchy

# COMMAND ----------

# DBTITLE 1,Markdown chunking
from langchain.text_splitter import MarkdownTextSplitter

markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
docs = markdown_splitter.create_documents([sample_text])

for i, doc in enumerate(docs):
    print(f"Chunk {i+1}: {doc}")

# COMMAND ----------

# DBTITLE 1,LaTex Chunking
from langchain.text_splitter import LatexTextSplitter
latex_splitter = LatexTextSplitter(chunk_size=100, chunk_overlap=0)
docs = latex_splitter.create_documents([sample_text])
for i, doc in enumerate(docs):
    print(f"Chunk {i+1}: {doc}")
