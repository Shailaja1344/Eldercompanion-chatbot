"""
Embedding generation and FAISS vector store management.
"""

from pathlib import Path
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR,
    FAISS_INDEX_NAME,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> HuggingFaceEmbeddings:
    """Load the HuggingFace sentence-transformer embedding model."""
    logger.info(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded successfully")
    return embeddings


def build_vector_store(
    chunks: list[Document],
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    save_path: Optional[Path] = None,
) -> FAISS:
    """
    Build a FAISS vector store from document chunks and persist it.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    save_path = save_path or FAISS_INDEX_DIR

    logger.info(f"Building FAISS index from {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Persist to disk
    vector_store.save_local(str(save_path), index_name=FAISS_INDEX_NAME)
    logger.info(f"FAISS index saved to {save_path}")

    return vector_store


def load_vector_store(
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    load_path: Optional[Path] = None,
) -> FAISS:
    """Load a persisted FAISS vector store from disk."""
    if embedding_model is None:
        embedding_model = get_embedding_model()

    load_path = load_path or FAISS_INDEX_DIR

    logger.info(f"Loading FAISS index from {load_path}")
    vector_store = FAISS.load_local(
        str(load_path),
        embedding_model,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded successfully")
    return vector_store


def rebuild_index(chunks: list[Document]) -> FAISS:
    """Rebuild the FAISS index from scratch."""
    logger.info("Rebuilding FAISS index from scratch...")
    embedding_model = get_embedding_model()
    return build_vector_store(chunks, embedding_model)
