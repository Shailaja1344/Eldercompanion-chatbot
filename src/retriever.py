"""
RAG retriever: similarity search over the FAISS vector store.
"""

from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import TOP_K_RETRIEVAL
from src.utils import setup_logger

logger = setup_logger(__name__)


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    domain_filter: Optional[str] = None,
) -> list[Document]:
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        vector_store: The FAISS vector store to search.
        query: User query string.
        top_k: Number of top results to return.
        domain_filter: Optional domain to filter results by.

    Returns:
        List of relevant Document chunks.
    """
    logger.info(f"Retrieving top-{top_k} chunks for: '{query[:80]}...'")

    if domain_filter:
        # Retrieve more and filter by domain
        results = vector_store.similarity_search(query, k=top_k * 3)
        filtered = [
            doc for doc in results
            if doc.metadata.get("domain") == domain_filter
        ]
        results = filtered[:top_k] if filtered else results[:top_k]
        logger.info(
            f"Domain filter '{domain_filter}': {len(filtered)} matches"
        )
    else:
        results = vector_store.similarity_search(query, k=top_k)

    logger.info(f"Retrieved {len(results)} chunks")
    return results


def retrieve_with_scores(
    vector_store: FAISS,
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[tuple[Document, float]]:
    """
    Retrieve chunks with their similarity scores.
    Lower score = more similar.
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)
    for doc, score in results:
        logger.debug(
            f"Score: {score:.4f} | Domain: {doc.metadata.get('domain')}"
        )
    return results


def format_context(chunks: list[Document]) -> str:
    """Combine retrieved chunks into a single context string."""
    if not chunks:
        return "No relevant information found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        domain = chunk.metadata.get("label", "General")
        context_parts.append(
            f"[Source: {domain}]\n{chunk.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)
