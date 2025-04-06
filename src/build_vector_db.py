import os
from dotenv import load_dotenv
from src.utils.document_processor import DocumentProcessor
from src.utils.vector_store import VectorStore

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
    # Check if DB exists and we're not forcing rebuild
    index_file = os.path.join(output_dir, "pdf_docs.faiss")
    docstore_file = os.path.join(output_dir, "pdf_docs.pkl")

    if os.path.exists(index_file) and os.path.exists(docstore_file) and not force:
        print(f"Vector database already exists at {output_dir}")
        return False

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist")
        return False

    # Count how many PDFs in the directory
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in {data_dir}:")
    for pdf in pdf_files:
        print(f"  - {pdf}")

    try:
        # Initialize components
        doc_processor = DocumentProcessor(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        print(f"Processing documents in {data_dir}")

        vector_store = VectorStore(persist_directory=output_dir)
        print(f"Vector store initialized with {output_dir}")

        # Process documents and add to vector store
        print(f"Processing documents from {data_dir}...")
        documents = doc_processor.process_directory(data_dir)

        # Check if documents list is empty
        if not documents:
            print("Error: No documents were processed. Check the PDF files and logs.")
            return False

        print(f"Successfully processed {len(documents)} document chunks")

        # Add documents to vector store
        print(f"Adding documents to vector store...")
        vector_store.add_documents(documents)

        print(f"Vector database built successfully with {len(documents)} chunks")
        return True
    except Exception as e:
        print(f"Error building vector database: {e}")
        import traceback

        traceback.print_exc()
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
        print("Vector database build completed successfully.")
    else:
        print("Vector database build failed or was skipped.")
