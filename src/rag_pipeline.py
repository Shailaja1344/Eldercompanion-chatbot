"""
End-to-end RAG pipeline orchestration.
Combines domain classification, retrieval, and LLM generation.
"""

from typing import Optional
from pathlib import Path

from src.config import FAISS_INDEX_DIR
from src.data_loader import load_and_chunk_all
from src.embeddings import (
    get_embedding_model,
    build_vector_store,
    load_vector_store,
)
from src.retriever import retrieve_relevant_chunks, format_context
from src.llm_engine import load_llm, generate_response
from src.domain_classifier import classify_domain, get_domain_info
from src.utils import setup_logger, is_emergency_query

logger = setup_logger(__name__)


class SeniorChatbotPipeline:
    """
    Main RAG pipeline for the Senior Citizen Chatbot.
    Manages the full flow: classify → retrieve → generate.
    """

    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.llm = None
        self.is_initialized = False
        self.conversation_history = []

    def initialize(self, rebuild_index: bool = False):
        """Load all models and the vector store."""
        logger.info("Initializing Senior Chatbot Pipeline...")

        # 1. Load embedding model
        logger.info("Step 1/3: Loading embedding model...")
        self.embedding_model = get_embedding_model()

        # 2. Load or build vector store
        logger.info("Step 2/3: Setting up vector store...")
        index_file = FAISS_INDEX_DIR / "senior_chatbot_index.faiss"

        if rebuild_index or not index_file.exists():
            logger.info("Building new FAISS index...")
            chunks = load_and_chunk_all()
            self.vector_store = build_vector_store(
                chunks, self.embedding_model
            )
        else:
            self.vector_store = load_vector_store(self.embedding_model)

        # 3. Load LLM
        logger.info("Step 3/3: Loading LLM...")
        self.llm = load_llm()

        self.is_initialized = True
        logger.info("Pipeline initialized successfully!")

    def query(
        self,
        user_query: str,
        domain_override: Optional[str] = None,
    ) -> dict:
        """
        Process a user query through the full RAG pipeline.

        Returns dict with: response, domain, domain_info, is_emergency,
                          sources, confidence
        """
        if not self.is_initialized:
            self.initialize()

        # Simple greeting shortcut
        greeting = user_query.strip().lower()
        if greeting in {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}:
            # Return a friendly greeting without invoking the RAG pipeline
            return {
                "response": "Hello! I'm here to help you. How can I assist you today?",
                "domain": "companion",
                "domain_info": get_domain_info("companion"),
                "is_emergency": False,
                "sources": [],
                "confidence": 1.0,
            }

        # Handle very short or vague queries
        if len(user_query.strip()) < 4:
            return {
                "response": "I see you've typed a short message. Could you please tell me a bit more so I can help you better? For example, you can ask 'How do I make a phone call?'",
                "domain": "companion",
                "domain_info": get_domain_info("companion"),
                "is_emergency": False,
                "sources": [],
                "confidence": 0.0,
            }

        # Check for emergency
        emergency = is_emergency_query(user_query)
        if emergency:
            emergency_prefix = (
                "⚠️ **If this is a life-threatening emergency, "
                "please call 911 immediately.**\n\n"
            )
        else:
            emergency_prefix = ""

        # Classify domain
        if domain_override:
            domain = domain_override
            confidence = 1.0
        else:
            domain, confidence = classify_domain(
                user_query,
                self.embedding_model,
                self.vector_store,
            )

        domain_info = get_domain_info(domain)

        # Retrieve relevant chunks
        chunks = retrieve_relevant_chunks(
            self.vector_store,
            user_query,
            domain_filter=domain if confidence > 0.05 else None,
        )
        # If no useful chunks, fall back to LLM-only generation
        if not chunks:
            logger.info("No relevant chunks found; using LLM without context.")
            context = ""
        else:
            context = format_context(chunks)

        # Generate response
        response = generate_response(self.llm, context, user_query)
        full_response = emergency_prefix + response

        # Track sources
        sources = list(set(
            c.metadata.get("source", "unknown") for c in chunks
        ))

        # Store in history
        self.conversation_history.append({
            "query": user_query,
            "response": full_response,
            "domain": domain,
        })

        result = {
            "response": full_response,
            "domain": domain,
            "domain_info": domain_info,
            "is_emergency": emergency,
            "sources": sources,
            "confidence": confidence,
        }

        logger.info(
            f"Query processed — domain: {domain}, "
            f"emergency: {emergency}, sources: {sources}"
        )
        return result

    def get_history(self) -> list[dict]:
        """Return conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def rebuild_index(self):
        """Rebuild the FAISS index from scratch."""
        logger.info("Rebuilding index...")
        chunks = load_and_chunk_all()
        self.vector_store = build_vector_store(
            chunks, self.embedding_model
        )
        logger.info("Index rebuilt successfully")
