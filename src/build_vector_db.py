import os
import shutil
import traceback
from dotenv import load_dotenv
from src.utils.document_processor import DocumentProcessor
from src.utils.vector_store import VectorStore
from src.utils.logger import get_logger

# Get logger
logger = get_logger(__name__)

load_dotenv()


def build_vector_database(
    data_dir="data/raw/pdf",
    output_dir="data/faiss_index",
    force=False,
    chunk_size=1000,
    chunk_overlap=200,
):
    """
    Parse documents in raw data directory and create a FAISS vector store.

    Args:
        data_dir: Directory containing raw documents
        output_dir: Directory to store the vector database
        force: Whether to rebuild database if it exists
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting vector database build...")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Force rebuild: {force}")

    # Check if DB exists and we're not forcing rebuild
    index_file = os.path.join(output_dir, "pdf_docs.faiss")
    docstore_file = os.path.join(output_dir, "pdf_docs.pkl")

    # If force=True, delete existing files
    if force and os.path.exists(output_dir):
        logger.info(
            f"Force rebuild requested. Removing existing vector database at {output_dir}"
        )
        if os.path.exists(index_file):
            os.remove(index_file)
            logger.info(f"Removed {index_file}")
        if os.path.exists(docstore_file):
            os.remove(docstore_file)
            logger.info(f"Removed {docstore_file}")
    elif os.path.exists(index_file) and os.path.exists(docstore_file) and not force:
        logger.info(f"Vector database already exists at {output_dir}")
        return False

    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Error: Data directory {data_dir} does not exist")
        return False

    # Count how many PDFs in the directory
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}:")
    for pdf in pdf_files:
        logger.info(f"  - {pdf}")

    try:
        # Initialize components
        doc_processor = DocumentProcessor(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logger.info(
            f"Document processor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

        # Make sure the output_dir exists
        os.makedirs(output_dir, exist_ok=True)

        vector_store = VectorStore(persist_directory=output_dir)
        logger.info(f"Vector store initialized with {output_dir}")

        # Process documents and add to vector store
        logger.info(f"Processing documents from {data_dir}...")
        documents = doc_processor.process_directory(data_dir)

        # Check if documents list is empty
        if not documents:
            logger.error(
                "Error: No documents were processed. Check the PDF files and logs."
            )
            return False

        logger.info(f"Successfully processed {len(documents)} document chunks")

        # Add documents to vector store
        logger.info(f"Adding documents to vector store...")
        vector_store.add_documents(documents)

        logger.info(f"Vector database built successfully with {len(documents)} chunks")
        return True
    except Exception as e:
        logger.error(f"Error building vector database: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Use default parameters
    result = build_vector_database(
        data_dir="data/raw/pdf",
        output_dir="data/faiss_index",
        force=True,  # Change to True to force rebuild
        chunk_size=1000,
        chunk_overlap=200,
    )

    if result:
        logger.info("Vector database build completed successfully.")
    else:
        logger.warning("Vector database build failed or was skipped.")
