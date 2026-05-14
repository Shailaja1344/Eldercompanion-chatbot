"""
LLM engine: loads HuggingFace model and generates responses.
"""

from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
)
from src.utils import setup_logger, format_response_for_seniors

logger = setup_logger(__name__)

# Module-level cache for the LLM
_llm_cache = None


def load_llm(
    model_name: str = LLM_MODEL_NAME,
    temperature: float = LLM_TEMPERATURE,
) -> ChatGroq:
    """
    Load the Groq LLM.
    Uses module-level caching to avoid reloading.
    """
    global _llm_cache
    if _llm_cache is not None:
        logger.info("Using cached LLM instance")
        return _llm_cache

    logger.info(f"Loading LLM via Groq: {model_name}")
    
    llm = ChatGroq(
        temperature=temperature,
        model_name=model_name,
    )

    _llm_cache = llm
    logger.info("LLM loaded successfully")
    return llm


def get_prompt_template() -> PromptTemplate:
    """Create the senior-friendly prompt template."""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT,
    )


def generate_response(
    llm: ChatGroq,
    context: str,
    question: str,
) -> str:
    """
    Generate a response using the LLM with retrieved context.
    """
    prompt_template = get_prompt_template()
    # Build LCEL chain: prompt -> LLM -> output parser
    chain = get_prompt_template() | llm | StrOutputParser()
    try:
        logger.info(f"Generating response for: '{question[:80]}'")
        raw_response = chain.invoke({"context": context, "question": question})
        response = format_response_for_seniors(raw_response)
        logger.info(f"Response generated ({len(response)} chars)")
        return response
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return (
            "I'm sorry, I had trouble understanding that. "
            "Could you please try asking in a different way?"
        )


def clear_llm_cache():
    """Clear the cached LLM instance."""
    global _llm_cache
    _llm_cache = None
    logger.info("LLM cache cleared")
