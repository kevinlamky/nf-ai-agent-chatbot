import os
import pymupdf
import pandas as pd
from typing import List, Dict, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """A class for processing various document types (PDF, CSV) for the AI agent."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file and split into chunks.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of document chunks with metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Extract text from PDF
        doc = pymupdf.open(file_path)
        text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()

        # Split text into chunks
        chunks = self.text_splitter.create_documents([text])

        # Add metadata to chunks
        file_name = os.path.basename(file_path)
        documents = []

        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "content": chunk.page_content,
                    "metadata": {
                        "source": file_name,
                        "chunk": i,
                        "file_path": file_path,
                    },
                }
            )

        return documents

    def process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a CSV file into documents.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of documents with metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        documents = []

        # Convert each row to a document
        for index, row in df.iterrows():
            content = " ".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(
                {
                    "content": content,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "index": index,
                        "file_path": file_path,
                    },
                }
            )

        return documents

    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.

        Args:
            dir_path: Path to the directory

        Returns:
            List of all documents with metadata
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        documents = []

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)

            if file_name.lower().endswith(".pdf"):
                documents.extend(self.process_pdf(file_path))
            elif file_name.lower().endswith(".csv"):
                documents.extend(self.process_csv(file_path))

        print(f"Processed {len(documents)} documents from {dir_path}")

        return documents


if __name__ == "__main__":
    processor = DocumentProcessor()

    # Test processing a single PDF document
    pdf_path = "data/raw/pdf/1_Undershaft_Planning_Statement.pdf"
    pdf_documents = processor.process_pdf(pdf_path)
    print(f"Processed {len(pdf_documents)} chunks from {pdf_path}")
    print(pdf_documents[0])

    # Test processing a single CSV document
    csv_path = "data/raw/csv/Planning Application Details.csv"
    csv_documents = processor.process_csv(csv_path)
    print(f"Processed {len(csv_documents)} rows from {csv_path}")
    print(csv_documents[0])

    # Test processing a directory
    test_dir = "data/raw/pdf"
    dir_documents = processor.process_directory(test_dir)
    print(
        f"Processed {len(set(doc['metadata']['source'] for doc in dir_documents))} total documents from directory {test_dir}"
    )
    print(
        f"Document types: {set(doc['metadata']['source'].split('.')[-1] for doc in dir_documents)}"
    )
