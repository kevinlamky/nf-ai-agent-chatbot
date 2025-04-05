import os
import pandas as pd
from dotenv import load_dotenv

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import (
    create_csv_agent,
    create_pandas_dataframe_agent,
)
from langchain_openai import AzureOpenAI, AzureChatOpenAI

load_dotenv()

# Fix SSL certificate verification issues
os.environ.pop("SSL_CERT_FILE", None)  # Remove problematic SSL_CERT_FILE if it exists

# Use AzureChatOpenAI instead of AzureOpenAI for function calling
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
)

agent = create_csv_agent(
    llm,
    "data/raw/csv/Planning Application Details.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
    number_of_head_rows=2,
)

agent.invoke(
    "What is the current status of the planning application on New Brent Street?"
)
agent.invoke(
    "What details are available about the planning application on New Brent Street?"
)
agent.invoke(
    "How many planning applications have been approved in the London Borough of Barnet?"
)
