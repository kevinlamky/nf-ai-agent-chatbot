"""
Prompt templates for document processing and vector search.
"""

# Prompts for document search results
DOCUMENT_SEARCH_RESULTS_PREFIX = (
    "Here is relevant information from the planning documents:\n\n"
)
DOCUMENT_SEARCH_NO_RESULTS = "No relevant information found in the planning documents."
DOCUMENT_SEARCH_NOT_AVAILABLE = "PDF document search is not available."
DOCUMENT_SEARCH_ERROR = "Error searching documents: {error}"

# Format for document results
DOCUMENT_RESULT_FORMAT = "Document: {source}\n{content}"

# CSV search messages
CSV_SEARCH_NOT_AVAILABLE = "CSV data search is not available."
CSV_SEARCH_ERROR = "Error querying CSV data: {error}"
