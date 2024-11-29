# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is fourth in a series that **takes in synthetic data for Fine Tuning (FT)**.
# MAGIC
# MAGIC What this notebook does:
# MAGIC 1. Perform fine tuning using a chat model on the synthetic data generated and prepared in NBs 1-3
# MAGIC 2. Serve the model on an endpoint
# MAGIC 3. Perform inference using the endpoint

# COMMAND ----------

# MAGIC %pip install -U databricks-genai databricks-sdk mlflow databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip freeze
# MAGIC # mlflow==2.16.2

# COMMAND ----------

from langchain_core.prompts import PromptTemplate
from random import sample
from databricks_langchain import ChatDatabricks
import mlflow
from mlflow.langchain.output_parsers import ChatCompletionsOutputParser

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set parameters and names

# COMMAND ----------

catalog = "yen"
db = "syn_data_gen"

train_table_name = f"{catalog}.{db}.train"
test_table_name = f"{catalog}.{db}.test"

base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ft_model_name = f"{catalog}.{db}.pubmed_rag_model"

base_endpoint_name = "databricks-meta-llama-3-1-70b-instruct"
model_endpoint_name = "pubmed_rag_model"
inference_table_name = model_endpoint_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up chain with approprimate prompt, llm and output_parser

# COMMAND ----------

llm = ChatDatabricks(endpoint=model_endpoint_name)
output_parser = ChatCompletionsOutputParser()

chain = llm | output_parser

# COMMAND ----------

input = [{'role': 'system',
  'content': 'You are a medical expert answering questions about biomedical research. Please answer the question below based on only the provided context. If you do not know, return nothing.'},
 {'role': 'user',
  'content': 'What is the primary objective of cancer immunotherapy?. Answer this question using only this context: \naspire to increase the overall knowledge of breast cancer and improve outcomes\nthrough proactive health practices.\n\nAlthough our study is the first in the Northern Border region of Saudi Arabia\nto conduct such a survey and to use a pre-validated tool to collect the\nspecified data, potentially yielding valuable insights, it is not without\nlimitations. The study\'s cross-sectional nature restricts our ability to infer\ncausation, and the reliance on self-reporting through questionnaires may\nintroduce response biases as only people with internet access and/or social\nmedia presence will respond to this survey. Also, using the convenience\nsampling method for participant recruitment may introduce certain biases, such\nas "selection" or "nonresponse" biases, that should be considered in future\nstudies.\n\nConclusions\n\nThis study highlights a spectrum of awareness and engagement in breast health\nbehaviors among women in Saudi Arabia\'s Northern Border region. The\nparticipants displayed a commendable understanding of certain breast cancer\nsigns, such as the presence of lumps, and recognized lactation as a preventive\nmeasure against the disease. Despite these positives, there are notable\nknowledge deficits concerning additional signs, symptoms, risk factors, and\npreventive practices.\n\nAlarmingly, the majority of study participants have not engaged in either\nclinical breast examinations or mammography. These findings underscore an\nurgent need for enhanced health education initiatives tailored to women. There\nis a critical mandate to motivate regular participation in breast cancer\nscreenings and to dispel myths surrounding mammography, particularly among\nthose with a familial predisposition to the condition, who would benefit from\nannual screenings.\n\n.'}]


# COMMAND ----------

chain.invoke(input)

# COMMAND ----------

# prompt_template = """
# You are a medical expert answering questions about biomedical research. Please answer the question below based on only the provided context. If you do not know, return nothing. 

# Question: What is the effect of PH extracts on the proliferation of cancer cells?. Answer this question using only the context if given.
# """

# prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
# chain = prompt | llm | output_parser
# chain.invoke({"question": "what causes breast cancer?"})

# COMMAND ----------

# ensure that langchain is < 0.3.0
mlflow.langchain.autolog()
mlflow.models.set_model(chain)
