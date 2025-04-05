import os
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Fix SSL certificate verification issues
os.environ.pop("SSL_CERT_FILE", None)  # Remove problematic SSL_CERT_FILE if it exists

llm = AzureOpenAI(
    api_token=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
)

df = SmartDataframe(
    "data/raw/csv/Planning Application Details.csv", config={"llm": llm}
)

df.chat("What is the current status of the planning application on CHEQUERS LANE?")
