# Planning Applications AI Assistant - Streamlit UI

This is a simple Streamlit UI for the Planning Applications AI Agent. It provides a user-friendly interface to interact with the AI assistant that answers questions about planning applications in London.

## Features

- Clean chat interface with user and assistant messages
- Sidebar with example questions for quick testing
- Conversation history management
- Responsive design
- Clear conversation button

## Setup

1. Make sure you have all dependencies installed:

```bash
poetry install
```

2. Set up your environment variables in the `.env` file:

```
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_CHAT_MODEL=your_model_name
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_EMBEDDING_MODEL=your_embedding_model

# Google Search API credentials (optional but recommended)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
```

3. Make sure the vector database is built (if using document search):

```bash
python src/build_vector_db.py
```

## Running the App

Run the Streamlit app with:

```bash
streamlit run app.py
```

The app will be available at http://localhost:8501 by default.

## Using the App

1. Type your question in the text area at the bottom of the page and click "Send"
2. The assistant will process your question and provide an answer
3. You can click on example questions in the sidebar to quickly test the agent
4. Use the "Clear Conversation" button to reset the conversation history

## Troubleshooting

- If you see an error about missing dependencies, make sure you've run `poetry install`
- If search doesn't work, check that your Google Search API credentials are properly set in the `.env` file
- If the agent can't answer document-specific questions, ensure you've built the vector database

## Dependencies

- streamlit
- openai
- langchain (with compatible components)
- pydantic
- google-api-python-client
- chromadb (for document search)
