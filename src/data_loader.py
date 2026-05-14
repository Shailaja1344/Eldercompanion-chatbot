"""
Document loader and text chunking for the RAG pipeline.
Loads domain-specific .txt files, splits them into chunks,
and tags each chunk with domain metadata.
"""

from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, DOMAINS
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_single_document(file_path: Path, domain: str) -> list[Document]:
    """Load a single .txt file and return as LangChain Documents."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []
    try:
        text = file_path.read_text(encoding="utf-8")
        doc = Document(
            page_content=text,
            metadata={
                "domain": domain,
                "source": file_path.name,
                "label": DOMAINS.get(domain, {}).get("label", domain),
            },
        )
        logger.info(f"Loaded {file_path.name} ({len(text)} chars)")
        return [doc]
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def load_all_documents(data_dir: Optional[Path] = None) -> list[Document]:
    """Load all domain .txt files from the data directory."""
    data_dir = data_dir or DATA_DIR
    all_docs = []
    for domain_key, domain_info in DOMAINS.items():
        file_path = data_dir / domain_info["file"]
        docs = load_single_document(file_path, domain_key)
        all_docs.extend(docs)
    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into smaller chunks for embedding and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} docs into {len(chunks)} chunks")
    return chunks


def load_and_chunk_all(
    data_dir: Optional[Path] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Load all documents and chunk them in one step."""
    documents = load_all_documents(data_dir)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    return chunks
