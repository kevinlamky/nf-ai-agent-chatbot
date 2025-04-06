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

load_dotenv()

logger = get_logger(__name__)

# Fix SSL certificate verification issues
os.environ.pop("SSL_CERT_FILE", None)


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

        # Initialize embedding model
        self._initialize_embeddings()

        # Set up vector store (load existing or create new)
        self._setup_vector_store()

    def _initialize_embeddings(self) -> None:
        """Initialize the Azure OpenAI embeddings model."""
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings model: {e}")
            raise

    def _setup_vector_store(self) -> None:
        """Set up the vector store by loading an existing index or creating a new one."""
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        if os.path.exists(self.index_file) and os.path.exists(self.docstore_file):
            try:
                self.vectordb = self._load_faiss_index()
                logger.info(f"Loaded existing FAISS index from {self.index_file}")
            except Exception as e:
                logger.error(f"Error loading existing FAISS index: {e}")
                self._initialise_index()
        else:
            logger.info("Creating new FAISS index")
            self._initialise_index()

    def _initialise_index(self) -> None:
        """Initialise an empty FAISS index with the correct embedding dimensions."""
        # Get embedding dimension from the embeddings model
        embedding_dim = len(self.embeddings.embed_query("test query"))

        # Create an empty index with the correct dimensions
        empty_index = faiss.IndexFlatL2(embedding_dim)

        # Create the FAISS vector store using the empty index
        self.vectordb = FAISS(
            embedding_function=self.embeddings,
            index=empty_index,
            docstore={},
            index_to_docstore_id={},
        )

        logger.info(f"Initialised empty FAISS index with dimension {embedding_dim}")

    def _load_faiss_index(self) -> FAISS:
        """Load an existing FAISS index from disk."""
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

        self._process_document_batches(batches, initial_backoff, max_retries)

    def _process_document_batches(
        self, batches: List[List[Document]], initial_backoff: int, max_retries: int
    ) -> None:
        """
        Process document batches with retry logic for rate limits.

        Args:
            batches: List of document batches to process
            initial_backoff: Initial backoff time in seconds
            max_retries: Maximum number of retry attempts
        """
        logger.info(f"Processing {len(batches)} batches of documents")

        for i, batch in enumerate(tqdm(batches, desc="Processing document batches")):
            retry_count = 0
            backoff_time = initial_backoff

            while retry_count <= max_retries:
                try:
                    # For the first batch, recreate the index if it's empty
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
                        raise

        # Save the index after processing all batches
        try:
            self._save_faiss_index()
            logger.info(f"Saved FAISS index to {self.persist_directory}")
        except Exception as e:
            logger.warning(f"Could not save FAISS index: {e}")

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
        self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None
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
        search_kwargs = search_kwargs or {"k": 5}
        return self.vectordb.as_retriever(search_kwargs=search_kwargs)
