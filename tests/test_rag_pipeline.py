"""Tests for the RAG pipeline module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.retriever import format_context
from src.utils import is_emergency_query, sanitize_text, format_response_for_seniors


class TestFormatContext:
    """Tests for context formatting."""

    def test_empty_chunks(self):
        result = format_context([])
        assert result == "No relevant information found."

    def test_single_chunk(self):
        chunks = [
            Document(
                page_content="Take medicine daily.",
                metadata={"label": "Healthcare"},
            )
        ]
        result = format_context(chunks)
        assert "Healthcare" in result
        assert "Take medicine daily." in result

    def test_multiple_chunks(self):
        chunks = [
            Document(
                page_content="Chunk 1",
                metadata={"label": "Healthcare"},
            ),
            Document(
                page_content="Chunk 2",
                metadata={"label": "Banking"},
            ),
        ]
        result = format_context(chunks)
        assert "Healthcare" in result
        assert "Banking" in result


class TestEmergencyDetection:
    """Tests for emergency query detection."""

    def test_911_query(self):
        assert is_emergency_query("Should I call 911?") is True

    def test_chest_pain(self):
        assert is_emergency_query("I have chest pain") is True

    def test_normal_query(self):
        assert is_emergency_query("How do I use WhatsApp?") is False

    def test_bleeding(self):
        assert is_emergency_query("I'm bleeding badly") is True

    def test_empty_query(self):
        assert is_emergency_query("") is False


class TestSanitizeText:
    """Tests for text sanitization."""

    def test_strips_whitespace(self):
        assert sanitize_text("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert sanitize_text("hello    world") == "hello world"

    def test_empty_string(self):
        assert sanitize_text("") == ""

    def test_none_input(self):
        assert sanitize_text(None) == ""


class TestFormatResponse:
    """Tests for senior-friendly response formatting."""

    def test_empty_response(self):
        result = format_response_for_seniors("")
        assert "try asking again" in result.lower()

    def test_normal_response(self):
        result = format_response_for_seniors("Take your medicine daily.")
        assert result == "Take your medicine daily."

    def test_excessive_newlines(self):
        result = format_response_for_seniors("Hello\n\n\n\n\nWorld")
        assert "\n\n\n" not in result
