"""
Shared utility functions: logging, text sanitization, formatting helpers.
"""

import logging
import re
from pathlib import Path

from src.config import LOG_LEVEL, LOG_FILE


def setup_logger(name: str = "senior_chatbot") -> logging.Logger:
    """
    Configure and return a named logger with both file and console handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid duplicate handlers

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def sanitize_text(text: str) -> str:
    """
    Clean and normalize user input text.
    - Strip whitespace
    - Remove excessive punctuation
    - Normalize spacing
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)  # Remove unusual characters
    return text


def format_response_for_seniors(text: str) -> str:
    """
    Format LLM output for better readability by senior citizens.
    - Ensure proper line breaks
    - Add spacing between sections
    - Clean up formatting artifacts
    - Strip internal prompt labels (Question/Helpful Answer)
    """
    if not text:
        return "I'm sorry, I couldn't generate a response. Please try asking again."

    # Remove prompt leakages if model repeats them
    # Specifically target the system instructions if they are echoed
    if "Follow these rules:" in text:
        text = text.split("Follow these rules:")[-1]
    
    if "Helpful Answer:" in text:
        text = text.split("Helpful Answer:")[-1]

    # Additional cleanup for specific prompt labels
    text = re.sub(r"^Question:.*?\n", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^Context:.*?\n", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^Helpful Answer:\s*", "", text, flags=re.IGNORECASE)
    
    # Remove any line that starts with "1. ", "2. ", etc. if it matches the instructions
    text = re.sub(r"^\d\.\s+(Use simple|Be warm|Give step|If the question|Keep answers|If you are|Always end).*", "", text, flags=re.MULTILINE | re.IGNORECASE)
    
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
    text = re.sub(r"  +", " ", text)         # Collapse double spaces

    # Ensure bullet points are properly formatted
    text = re.sub(r"^[-*]", "•", text, flags=re.MULTILINE)

    return text.strip()


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def get_domain_files(data_dir: Path) -> list:
    """Get all .txt files from the data directory."""
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.txt"))


def is_emergency_query(query: str) -> bool:
    """
    Quick check if a query might be about an emergency situation.
    Returns True if the query contains emergency-related keywords.
    """
    emergency_words = [
        "911", "emergency", "help me", "can't breathe",
        "chest pain", "heart attack", "stroke", "bleeding",
        "unconscious", "choking", "dying", "poison",
        "call ambulance", "fire", "i fell", "i'm hurt",
    ]
    query_lower = query.lower()
    return any(word in query_lower for word in emergency_words)
