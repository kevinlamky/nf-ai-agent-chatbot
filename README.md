# Planning Applications AI Assistant

An AI-powered chatbot that can answer questions about planning applications in London using PDF documents, CSV data, and web search capabilities.

## Features

- **PDF Document Search**: Uses RAG (Retrieval-Augmented Generation) with Azure OpenAI to search through planning application documents.
- **CSV Data Analysis**: Analyzes planning application data stored in CSV format.
- **Web Search**: Searches the web for additional information using Google Search API.
- **Interactive UI**: Simple Streamlit UI for chatting with the AI assistant.

## Requirements

- Python 3.9+
- Azure OpenAI API access
- Google Search API access (optional for web search capabilities)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nf-ai-agent-chatbot.git
cd nf-ai-agent-chatbot
```

2. Install dependencies:

Using Poetry:

```bash
poetry install
```

Using pip:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys and configurations:

```
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_CHAT_MODEL=your_chat_model_deployment_name
AZURE_OPENAI_EMBEDDING_MODEL=your_embedding_model_deployment_name

# Google Search (optional)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_custom_search_engine_id
```

## Data Setup

1. Place your PDF documents in the `data/raw/pdf/` directory
2. Place your CSV files in the `data/raw/csv/` directory

## Building the Vector Database

Before running the application, build the vector database for the PDF documents:

```bash
python -m src.build_vector_db
```

Optional arguments:

- `--data-dir`, `-d`: Directory containing PDF files (default: `data/raw/pdf`)
- `--output-dir`, `-o`: Directory to store the vector database (default: `data/chroma_db`)
- `--force`, `-f`: Force rebuild of database if it exists
- `--chunk-size`, `-s`: Size of document chunks (default: 1000)
- `--chunk-overlap`, `-c`: Overlap between chunks (default: 200)

## Running the Application

Start the Streamlit web interface:

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                   # Main Streamlit application
├── data/                    # Data directory
│   ├── raw/                 # Raw data files
│   │   ├── pdf/             # PDF documents
│   │   └── csv/             # CSV files
│   └── chroma_db/           # Vector database
├── src/                     # Source code
│   ├── agent/               # Agent components
│   │   ├── agent.py         # Main agent implementation
│   │   └── csv_agent.py     # CSV agent implementation
│   ├── utils/               # Utility functions
│   │   ├── document_processor.py  # Document processing
│   │   ├── search_tool.py         # Google Search integration
│   │   └── vector_store.py        # Vector store implementation
│   └── build_vector_db.py   # Script to build vector database
├── .env                     # Environment variables
└── README.md                # This file
```

## Deployment

For deployment instructions, see [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md).

## License

[MIT License](LICENSE)
