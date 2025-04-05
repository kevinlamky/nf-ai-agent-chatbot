import os
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# Fix SSL certificate verification issues
os.environ.pop("SSL_CERT_FILE", None)  # Remove problematic SSL_CERT_FILE if it exists


class VectorStore:
    """A vector store using ChromaDB with Azure OpenAI embeddings."""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "pdf_docs",
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection in ChromaDB
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        try:
            # Create embeddings model
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            )
        except Exception as e:
            print(f"Error creating embeddings model: {e}")

        # Create or load the vector store
        if os.path.exists(persist_directory):
            self.vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
        else:
            # Ensure directory exists
            os.makedirs(persist_directory, exist_ok=True)
            self.vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )

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

        for i, batch in enumerate(tqdm(batches, desc="Processing document batches")):
            retry_count = 0
            backoff_time = initial_backoff

            while retry_count <= max_retries:
                try:
                    # Add batch to vector store
                    self.vectordb.add_documents(batch)
                    # If successful, break out of retry loop
                    break

                except Exception as e:
                    error_message = str(e).lower()

                    # Check if it's a rate limit error
                    if (
                        "rate limit" in error_message
                        or "too many requests" in error_message
                    ):
                        retry_count += 1

                        if retry_count > max_retries:
                            raise Exception(f"Failed after {max_retries} retries: {e}")

                        print(
                            f"Rate limit hit. Retry {retry_count}/{max_retries}. Waiting {backoff_time}s..."
                        )
                        time.sleep(backoff_time)

                        # Exponential backoff
                        backoff_time *= 2
                    else:
                        # For other errors, just raise
                        raise e

        # Persist changes after all batches are processed
        self.vectordb.persist()

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
        return self.vectordb.similarity_search(query, k=k, filter=filter)

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
        return self.vectordb.similarity_search_with_score(query, k=k, filter=filter)

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

    def delete(self):
        """
        Delete all documents from the vector store.

        This method removes all documents from the vector database and persists the changes.
        """
        # Check if vectordb exists and has the delete_collection method
        if hasattr(self, "vectordb") and hasattr(self.vectordb, "delete_collection"):
            # Delete the collection
            self.vectordb.delete_collection()
            # Reinitialize the collection
            self.vectordb = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            # Persist the changes
            self.vectordb.persist()
            return True
        return False


if __name__ == "__main__":
    # test the vector store
    vector_store = VectorStore()

    vector_store.add_documents(
        [{"content": "Hello, world!", "metadata": {"source": "test.txt"}}]
    )

    documents = vector_store.similarity_search(
        "What is the planning statement for Undershaft?"
    )
    print(documents)

    documents = vector_store.similarity_search_with_score(
        "What is the planning statement for Undershaft?"
    )
    print(documents)

    vector_store.delete()
