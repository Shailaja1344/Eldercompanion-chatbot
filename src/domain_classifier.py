"""
Domain classifier: routes user queries to the appropriate domain.
Uses keyword matching + embedding similarity for robust classification.
"""

from typing import Optional

from src.config import DOMAINS
from src.utils import setup_logger

logger = setup_logger(__name__)


def classify_domain_keywords(query: str) -> tuple[str, float]:
    """
    Classify the domain of a query using keyword matching.

    Returns:
        Tuple of (domain_key, confidence_score).
    """
    query_lower = query.lower()
    scores = {}

    for domain_key, domain_info in DOMAINS.items():
        keywords = domain_info.get("keywords", [])
        matches = sum(1 for kw in keywords if kw in query_lower)
        if matches > 0:
            scores[domain_key] = matches / len(keywords)

    if not scores:
        logger.info("No keyword match — defaulting to 'companion'")
        return "companion", 0.0

    best_domain = max(scores, key=scores.get)
    confidence = scores[best_domain]
    logger.info(
        f"Domain classified: {best_domain} "
        f"(confidence: {confidence:.3f})"
    )
    return best_domain, confidence


def classify_domain_embedding(
    query: str,
    embedding_model=None,
    vector_store=None,
) -> tuple[str, float]:
    """
    Classify domain using embedding similarity against the vector store.
    Falls back to keyword classification if models aren't available.
    """
    if embedding_model is None or vector_store is None:
        return classify_domain_keywords(query)

    try:
        results = vector_store.similarity_search_with_score(query, k=3)
        if not results:
            return classify_domain_keywords(query)

        # Count domain votes from top results
        domain_votes = {}
        for doc, score in results:
            domain = doc.metadata.get("domain", "companion")
            if domain not in domain_votes:
                domain_votes[domain] = 0
            domain_votes[domain] += 1

        best_domain = max(domain_votes, key=domain_votes.get)
        confidence = domain_votes[best_domain] / len(results)

        logger.info(
            f"Embedding classification: {best_domain} "
            f"(confidence: {confidence:.3f})"
        )
        return best_domain, confidence

    except Exception as e:
        logger.warning(f"Embedding classification failed: {e}")
        return classify_domain_keywords(query)


def classify_domain(
    query: str,
    embedding_model=None,
    vector_store=None,
    prefer_embedding: bool = True,
) -> tuple[str, float]:
    """
    Main classification function. Combines keyword + embedding approaches.
    """
    kw_domain, kw_conf = classify_domain_keywords(query)

    if prefer_embedding and embedding_model and vector_store:
        emb_domain, emb_conf = classify_domain_embedding(
            query, embedding_model, vector_store
        )
        # Use embedding result if keyword confidence is low
        if kw_conf < 0.1 and emb_conf > 0.5:
            return emb_domain, emb_conf

    return kw_domain, kw_conf


def get_domain_info(domain_key: str) -> dict:
    """Get display info for a domain."""
    return DOMAINS.get(domain_key, {
        "icon": "💬",
        "label": "General",
        "description": "General assistance",
    })
