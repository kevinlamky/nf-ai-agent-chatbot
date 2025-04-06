import os
import time
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.utils.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

load_dotenv()

# Fix SSL certificate verification issues
os.environ.pop("SSL_CERT_FILE", None)  # Remove problematic SSL_CERT_FILE if it exists

# Check if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = "STREAMLIT_RUNTIME_PRODUCTION" in os.environ


class VectorStore:
    """A vector store using FAISS with Azure OpenAI embeddings."""

    def __init__(
        self,
        persist_directory: str = "data/faiss_index",
        index_name: str = "pdf_docs",
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the vector store
            index_name: Name of the FAISS index
        """
        self.persist_directory = persist_directory
        self.index_name = index_name
        self.index_file = os.path.join(persist_directory, f"{index_name}.faiss")
        self.docstore_file = os.path.join(persist_directory, f"{index_name}.pkl")

        try:
            # Create embeddings model
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            )
        except Exception as e:
            logger.error(f"Error creating embeddings model: {e}")
            raise

        # Create or load the vector store
        if os.path.exists(self.index_file) and os.path.exists(self.docstore_file):
            try:
                self.vectordb = self._load_faiss_index()
                logger.info(f"Loaded existing FAISS index from {self.index_file}")
            except Exception as e:
                logger.error(f"Error loading existing FAISS index: {e}")
                self._create_empty_index()
        else:
            # Create an empty index
            logger.info("Creating new FAISS index")
            self._create_empty_index()
            # Ensure directory exists
            os.makedirs(persist_directory, exist_ok=True)

    def _create_empty_index(self):
        """Create an empty FAISS index with a temporary dummy document."""
        # Create a dummy document for initialization
        dummy_doc = Document(
            page_content="Temporary initialization document",
            metadata={"source": "initialization", "temporary": True},
        )

        # Create the index with the dummy document
        self.vectordb = FAISS.from_documents([dummy_doc], self.embeddings)

        # Remove the dummy document from the index (this clears the internal docstore)
        try:
            self.vectordb.docstore._dict.clear()
            # Reset the index mapping
            self.vectordb.index_to_docstore_id = {}
            logger.info("Created empty FAISS index")
        except Exception as e:
            logger.warning(
                f"Warning: Couldn't fully clear dummy document from index: {e}"
            )

    def _load_faiss_index(self) -> FAISS:
        """Load a FAISS index from disk."""
        return FAISS.load_local(
            self.persist_directory,
            self.embeddings,
            self.index_name,
            allow_dangerous_deserialization=True,
        )

    def _save_faiss_index(self) -> None:
        """Save the FAISS index to disk."""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vectordb.save_local(self.persist_directory, self.index_name)

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 20,
        initial_backoff: int = 60,
        max_retries: int = 5,
    ) -> None:
        """
        Add documents to the vector store with reactive rate limit handling.

        Args:
            documents: List of documents with content and metadata
            batch_size: Number of documents to process in each batch
            initial_backoff: Initial wait time in seconds when hitting rate limits
            max_retries: Maximum number of retries per batch
        """
        # Check if documents list is empty
        if not documents:
            logger.warning("No documents to add to the vector store")
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        # Convert to LangChain document format
        langchain_docs = [
            Document(page_content=doc["content"], metadata=doc["metadata"])
            for doc in documents
        ]

        # Process in batches
        batches = [
            langchain_docs[i : i + batch_size]
            for i in range(0, len(langchain_docs), batch_size)
        ]

        logger.info(
            f"Processing documents in {len(batches)} batches of size {batch_size}"
        )

        for i, batch in enumerate(tqdm(batches, desc="Processing document batches")):
            retry_count = 0
            backoff_time = initial_backoff

            while retry_count <= max_retries:
                try:
                    # For the first batch, recreate the index if the index is empty (only contains dummy doc)
                    if (
                        i == 0
                        and retry_count == 0
                        and len(self.vectordb.index_to_docstore_id) == 0
                    ):
                        logger.info(
                            f"Creating new FAISS index with first batch of {len(batch)} documents"
                        )
                        self.vectordb = FAISS.from_documents(batch, self.embeddings)
                    else:
                        logger.debug(
                            f"Adding batch {i+1}/{len(batches)} ({len(batch)} documents)"
                        )
                        self.vectordb.add_documents(batch)

                    # If successful, break out of retry loop
                    break
                except Exception as e:
                    error_message = str(e).lower()
                    logger.error(f"Error in batch {i+1}: {e}")

                    # Check if it's a rate limit error
                    if (
                        "rate limit" in error_message
                        or "too many requests" in error_message
                    ):
                        retry_count += 1

                        if retry_count > max_retries:
                            raise Exception(f"Failed after {max_retries} retries: {e}")

                        logger.warning(
                            f"Rate limit hit. Retry {retry_count}/{max_retries}. Waiting {backoff_time}s..."
                        )
                        time.sleep(backoff_time)

                        # Exponential backoff
                        backoff_time *= 2
                    else:
                        # For other errors, just raise
                        raise e

        # Save the index after processing all batches
        try:
            self._save_faiss_index()
            logger.info(f"Saved FAISS index to {self.persist_directory}")
        except Exception as e:
            logger.warning(f"Warning: Could not save FAISS index: {e}")

    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents in the vector store.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of relevant documents
        """
        # Check if the index is empty
        if not self.vectordb.index_to_docstore_id:
            logger.warning("Vector store is empty. No results to return.")
            return []

        try:
            if filter:
                return self.vectordb.similarity_search(query, k=k, filter=filter)
            else:
                return self.vectordb.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def similarity_search_with_score(
        self, query: str, k: int = 3, filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents in the vector store with relevance scores.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of tuples containing document and score
        """
        # Check if the index is empty
        if not self.vectordb.index_to_docstore_id:
            logger.warning("Vector store is empty. No results to return.")
            return []

        try:
            if filter:
                return self.vectordb.similarity_search_with_score(
                    query, k=k, filter=filter
                )
            else:
                return self.vectordb.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search with score: {e}")
            return []

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever for the vector store.

        Args:
            search_kwargs: Search arguments

        Returns:
            Retriever interface for use with LangChain
        """
        search_kwargs = search_kwargs or {"k": 3}
        return self.vectordb.as_retriever(search_kwargs=search_kwargs)

    def delete(self):
        """
        Delete all documents from the vector store.

        This method removes all documents from the vector database and persists the changes.
        """
        try:
            # Create new empty index
            self._create_empty_index()

            # Save the empty index
            self._save_faiss_index()
            return True
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            return False


if __name__ == "__main__":
    logger.info("Vector store module loaded successfully")
    # Initialize the vector store
    vector_store = VectorStore()
    logger.info(
        f"Vector store contains {len(vector_store.vectordb.index_to_docstore_id)} documents"
    )

    documents = vector_store.similarity_search(
        "What is the planning statement for Undershaft?"
    )
    logger.info(documents)
