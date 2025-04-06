import os
import pymupdf
import pandas as pd
from typing import List, Dict, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger

logger = get_logger(__name__)


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

        Raises:
            FileNotFoundError: If the PDF file does not exist
        """
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            logger.info(f"Start processing PDF file: {file_path}")
            doc = pymupdf.open(file_path)
            text = ""

            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text += page.get_text()
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num}: {e}")

            if not text.strip():
                logger.warning(f"Warning: No text extracted from {file_path}")
                return []

            chunks = self.text_splitter.create_documents([text])
            file_name = os.path.basename(file_path)
            documents = []

            for i, chunk in enumerate(chunks):
                documents.append(
                    {
                        "content": chunk.page_content,
                        "metadata": {
                            "source": file_name,
                            "chunk": i,
                        },
                    }
                )

            logger.info(f"Processed {len(chunks)} chunks from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []

    # DEPRECATED: Use csv_agent instead
    def process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a CSV file into documents.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of documents with metadata

        Raises:
            FileNotFoundError: If the CSV file does not exist
        """
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            logger.info(f"Start processing CSV file: {file_path}")
            df = pd.read_csv(file_path)
            documents = []

            for index, row in df.iterrows():
                content = " ".join([f"{col}: {row[col]}" for col in df.columns])
                documents.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "index": index,
                        },
                    }
                )

            logger.info(f"Processed {len(documents)} rows from CSV {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            return []

    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.

        Args:
            dir_path: Path to the directory

        Returns:
            List of all documents with metadata

        Raises:
            FileNotFoundError: If the directory does not exist
        """
        if not os.path.exists(dir_path):
            logger.error(f"Directory not found: {dir_path}")
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        documents = []
        file_count = 0
        pdf_count = 0
        csv_count = 0

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            file_count += 1

            try:
                if file_name.lower().endswith(".pdf"):
                    pdf_count += 1
                    pdf_documents = self.process_pdf(file_path)
                    documents.extend(pdf_documents)
                    logger.info(
                        f"Extracted {len(pdf_documents)} chunks from {file_name}"
                    )
                elif file_name.lower().endswith(".csv"):
                    csv_count += 1
                    csv_documents = self.process_csv(file_path)
                    documents.extend(csv_documents)
                    logger.info(f"Extracted {len(csv_documents)} rows from {file_name}")
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")

        logger.info(f"Processed all documents from {dir_path}")
        logger.info(
            f"Total documents extracted: {file_count} ({pdf_count} PDF, {csv_count} CSV)"
        )

        return documents
