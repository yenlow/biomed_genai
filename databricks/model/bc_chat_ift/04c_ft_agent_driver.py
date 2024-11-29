# Databricks notebook source
# MAGIC %md
# MAGIC To set up the review app, we need to deploy the FT model as an agent with the Agent Framework.  Agent Framework only works with pyfunc or langchain flavors so models need to be wrapped as either.
# MAGIC
# MAGIC What this notebook does:
# MAGIC 1. Calls the [chain notebook](https://adb-830292400663869.9.azuredatabricks.net/editor/notebooks/1102774905469960?o=830292400663869#command/36116196036700) that wraps the finetuned model (`llm`) as a langchain chain.
# MAGIC ```
# MAGIC chain = llm | output_parser
# MAGIC ```
# MAGIC 2. Logs and registers the chain into mlflow
# MAGIC 3. Deploy the registered chain as an agent with built in review app

# COMMAND ----------

# MAGIC %pip install -U databricks-genai databricks-sdk databricks-langchain databricks-agents mlflow mlflow[databricks] 
# MAGIC #accelerate==0.33.0 torch==2.4.0 torchvision==0.19.0  transformers==4.43.4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip freeze
# MAGIC # mlflow==2.16.2

# COMMAND ----------

import os
import re
import pandas as pd
from databricks.sdk import WorkspaceClient
from langchain_core.prompts import PromptTemplate
from databricks_langchain import ChatDatabricks
from databricks import agents
import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.langchain.output_parsers import ChatCompletionsOutputParser
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import ChatCompletionRequest, ChatCompletionResponse
from pyspark.sql.functions import expr

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
ft_agent_name = f"{ft_model_name}_agent"

base_endpoint_name = "databricks-meta-llama-3-1-70b-instruct"
model_endpoint_name = "pubmed_rag_model"
inference_table_name = model_endpoint_name

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load sample input and output for chain signature

# COMMAND ----------

# Get dict/json input for a curl call
test = spark.table(test_table_name)
inputs = test.toPandas().to_dict(orient="records")
input = inputs[0]['messages'].tolist()[0:2]
input

# COMMAND ----------

output = {'choices': [{'index': 0,
   'message': {'role': 'assistant',
    'content': 'To prevent and treat cancer by harnessing the power of the immune system to target and destroy cancer cells.'},
   'finish_reason': 'stop'}],
 'object': 'chat.completion'}

# COMMAND ----------

signature = ModelSignature(ChatCompletionRequest(), ChatCompletionResponse())
signature

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log chain into mlflow
# MAGIC This converts code into a model artifact

# COMMAND ----------

with mlflow.start_run() as run:
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), '04b_ft_agent_chain'),
        resources=[DatabricksServingEndpoint(endpoint_name=model_endpoint_name)],
#        pip_requirements="requirements.txt",
        artifact_path='model',
        input_example=input,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## [OPTIONAL] Evaluate model as an agent
# MAGIC This is not necessary as the model is already evaluated in [05_eval_llm](https://adb-830292400663869.9.azuredatabricks.net/editor/notebooks/3995522635317170?o=830292400663869).<br>
# MAGIC Wrap served model into a mlflow client (see [doc](https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluate-agent.html#option-4-local-function-in-the-notebook))

# COMMAND ----------

def model(query):
    client = mlflow.deployments.get_deploy_client("databricks")
    return client.predict(endpoint=f"endpoints:/{model_endpoint_name}",
                          inputs={"messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]})

# COMMAND ----------

evaluation_results = mlflow.evaluate(
    data=eval_dataset,  # pandas DataFrame with just the evaluation set
    model = model,
    model_type="databricks-agent"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register chain

# COMMAND ----------

run_id = logged_agent_info.run_id
#run_id = "1e576df12cb94a4eac6f80c3f8e7469e"
model_uri = f"runs:/{run_id}/model"
model_uri

# COMMAND ----------

# register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=model_uri, name=ft_agent_name)

# COMMAND ----------

agents.deploy(ft_agent_name, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative (not recommended): Make the finetuned model an agent via a UC function
# MAGIC Ref: 
# MAGIC - https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html#assign-unity-catalog-tools-to-agents
# MAGIC - https://databricks.atlassian.net/browse/ES-1285830
