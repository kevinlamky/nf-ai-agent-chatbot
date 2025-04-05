import os
from dotenv import load_dotenv

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import AzureChatOpenAI
from src.prompts.csv_agent_prompts import CSV_AGENT_PREFIX

load_dotenv()

# Fix SSL certificate verification issues
os.environ.pop("SSL_CERT_FILE", None)


def get_csv_agent(csv_file_path, verbose=True, number_of_head_rows=2):
    """
    Create and return a CSV agent using Azure OpenAI.

    Args:
        csv_file_path: Path to the CSV file
        verbose: Whether to enable verbose output
        number_of_head_rows: Number of rows to display in head preview

    Returns:
        CSV agent
    """
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
    )

    agent = create_csv_agent(
        llm,
        csv_file_path,
        verbose=verbose,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        number_of_head_rows=number_of_head_rows,
        prefix=CSV_AGENT_PREFIX,
    )

    return agent
