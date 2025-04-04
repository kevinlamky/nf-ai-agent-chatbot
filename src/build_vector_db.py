#!/usr/bin/env python3

import os
import argparse
from dotenv import load_dotenv
from src.utils.document_processor import DocumentProcessor
from src.utils.vector_store import VectorStore

load_dotenv()


def build_vector_database(
    data_dir="data/raw",
    output_dir="data/chroma_db",
    force=False,
    chunk_size=1000,
    chunk_overlap=200,
):
    """
    Parse documents in raw data directory and create a ChromaDB vector store.

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
    if os.path.exists(output_dir) and not force:
        print(f"Vector database already exists at {output_dir}")
        return False

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist")
        return False

    try:
        # Initialize components
        doc_processor = DocumentProcessor(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        vector_store = VectorStore(persist_directory=output_dir)

        # Process documents and add to vector store
        documents = doc_processor.process_directory(data_dir)
        vector_store.add_documents(documents)

        print(f"Vector database built successfully with {len(documents)} chunks")
        return True
    except Exception as e:
        print(f"Error building vector database: {e}")
        return False


if __name__ == "__main__":
    build_vector_database()
