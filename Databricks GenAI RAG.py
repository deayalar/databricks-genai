# Databricks notebook source
# MAGIC %pip install --quiet PyPDF2 llama-index transformers databricks-sdk mlflow
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import requests
from pathlib import Path

# COMMAND ----------

DATA_CATALOG = "david_ayala_1olo_da"
VOLUME_NAME = "remote-files"
VOLUME_PATH = f"/Volumes/{DATA_CATALOG}/default/{VOLUME_NAME}"
VECTOR_SEARCH_ENDPOINT = "vector_search_endpoint"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Source Files
# MAGIC
# MAGIC The goal is to download the files from the remote source and then save them in the Unity Catalog as elements of a Volume
# MAGIC https://docs.databricks.com/en/ingestion/file-upload/download-internet-files.html#download-a-file-to-a-volume

# COMMAND ----------

# Create Unity Catalog data
spark.sql(f"CREATE CATALOG IF NOT EXISTS {DATA_CATALOG}")
# spark.sql(f"CREATE VOLUME {DATA_CATALOG}.default.{VOLUME_NAME}")

# COMMAND ----------

docs = [
  {
    "source": "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",
    "name": "attention.pdf"
  },
  {
    "source": "https://openreview.net/pdf?id=e2TBb5y0yFf",
    "name": "llms-reasoners.pdf"
  }
]

# COMMAND ----------

for doc in docs:
    response = requests.get(doc["source"])
    if response.status_code == 200:
        with open(Path(VOLUME_PATH, doc["name"]), "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download {doc['name']}. Status code: {response.status_code}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store Documents in a Table
# MAGIC
# MAGIC - Load raw files from volume to a dataframe
# MAGIC - Save raw files as table in Unity Catalog
# MAGIC

# COMMAND ----------

table_name = f"{DATA_CATALOG}.default.lab_pdf_raw_text"

# COMMAND ----------

# Load documents
df = spark.read.format("binaryFile").options(mimeType="pdf/*",
                                             recursiveFileLookup=True).load(VOLUME_PATH)

# Save raw df as table
df.write.mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

df.show()

# COMMAND ----------

type(df)

# COMMAND ----------

import PyPDF2
def get_text(file):
    with open(file, mode="rb") as file:
        reader = PyPDF2.PdfReader(file)
        # print("Number of pages:", len(reader.pages))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        # print(text)
        return text

# COMMAND ----------

text = get_text("/Volumes/david_ayala_1olo_da/default/remote-files/attention.pdf")
print(text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split Text

# COMMAND ----------

import pandas as pd
from typing import Iterator
from pyspark.sql.functions import col, pandas_udf, explode
from pyspark.sql.types import ArrayType, StringType, LongType
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import set_global_tokenizer
from llama_index.core import Document
from transformers import AutoTokenizer

@pandas_udf(returnType=ArrayType(StringType()))
def get_chunks(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    node_parser = SentenceSplitter(chunk_size=512, 
                                   chunk_overlap=20)
    def split_and_chunk(path: str) -> list[str]:
        path = path.replace('dbfs:', '')
        text = get_text(path)
        nodes = node_parser.get_nodes_from_documents([Document(text=text)])
        return [node.text for node in nodes]

    for batch in batch_iter:
        yield batch.apply(split_and_chunk)         

# COMMAND ----------

# MAGIC %md
# MAGIC Apply the pandas UDF Function to the dataframe

# COMMAND ----------

df_chunks = (df
                .withColumn("content", explode(get_chunks("path")))
                .selectExpr('path as pdf_name', 'content')
                )
display(df_chunks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Embeddings using Databricks Hosted Model
# MAGIC
# MAGIC Query Foundation models
# MAGIC https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#language-Databricks%C2%A0Python%C2%A0SDK

# COMMAND ----------

import time

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 25
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())
        time.sleep(2)
    return pd.Series(all_embeddings)

# COMMAND ----------

df_chunk_emd = (df_chunks
                .withColumn("embedding", get_embedding("content"))
                .selectExpr('pdf_name', 'content', 'embedding')
                )
display(df_chunk_emd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save dataframe with embeddings as table

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG 'david_ayala_1olo_da';
# MAGIC CREATE TABLE IF NOT EXISTS lab_pdf_text_embeddings (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   pdf_name STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC   -- Note: the table has to be CDC because VectorSearch is using DLT that is requiring CDC state
# MAGIC   ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

embedding_table_name = f"{DATA_CATALOG}.default.lab_pdf_text_embeddings"
df_chunk_emd.write.mode("append").saveAsTable(embedding_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search

# COMMAND ----------

# MAGIC %md
# MAGIC Go to Compute -> Vector Search -> Create Endpoint with type Standard use the name in the variable VECTOR_SEARCH_ENDPOINT

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain databricks-vectorsearch flashrank databricks-sdk langchain-community
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c


vector_search_client = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Index

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
  try:
      dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
      return dict_vsindex.get('status').get('ready', False)
  except Exception as e:
      if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
          print(f'Unexpected error describing the index. This could be a permission issue.')
          raise e
  return False

# COMMAND ----------

# Where we want to store our index
vs_index_fullname = f"{DATA_CATALOG}.default.pdf_text_self_managed_vs_index"

# COMMAND ----------

# After vector search endpoint is ready, create the index and sync it with the table that contains the embeddings

# The table we'd like to index
source_table_fullname = f"{DATA_CATALOG}.default.lab_pdf_text_embeddings"

# create or sync the index
if not index_exists(vector_search_client, VECTOR_SEARCH_ENDPOINT, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT}...")
  vector_search_client.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding"
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vector_search_client.get_index(VECTOR_SEARCH_ENDPOINT, vs_index_fullname).sync()

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAG Application

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings


# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
print(f"Test embeddings: {embedding_model.embed_query('What is the attention mechanism?')[:20]}...")

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=vs_index_fullname
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    # k defines the top k documents to retrieve
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.invoke("What is the advantage of Attention compared to other approaches?")
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

from langchain.chat_models import ChatDatabricks


# Test Databricks Foundation LLM model
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 300)
print(f"Test chat model: {chat_model.invoke('What is the attention mechanism?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC Prompt taken from the langchain hub: https://smith.langchain.com/hub/rlm/rag-prompt

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks


TEMPLATE = """You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:

<context>
{context}
</context>

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

question = {"query": "What is the advantage of attention compared to RNN?"}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the model to model registry in Unity Catalog

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain


# set model registery to UC
mlflow.set_registry_uri("databricks-uc")
model_name = f"{DATA_CATALOG}.default.rag_app"

with mlflow.start_run(run_name="rag_app_demo") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever, 
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )
