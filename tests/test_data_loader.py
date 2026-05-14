"""Tests for the data loader module."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.config import DOMAINS, DATA_DIR
from src.data_loader import (
    load_single_document,
    load_all_documents,
    chunk_documents,
    load_and_chunk_all,
)


class TestLoadSingleDocument:
    """Tests for loading individual documents."""

    def test_load_existing_file(self):
        """Test loading an existing healthcare file."""
        file_path = DATA_DIR / "healthcare.txt"
        docs = load_single_document(file_path, "healthcare")
        assert len(docs) == 1
        assert docs[0].metadata["domain"] == "healthcare"
        assert docs[0].metadata["source"] == "healthcare.txt"
        assert len(docs[0].page_content) > 0

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        file_path = DATA_DIR / "nonexistent.txt"
        docs = load_single_document(file_path, "unknown")
        assert len(docs) == 0

    def test_metadata_has_label(self):
        """Test that loaded docs have domain label metadata."""
        file_path = DATA_DIR / "banking.txt"
        docs = load_single_document(file_path, "banking")
        assert docs[0].metadata["label"] == "Banking"


class TestLoadAllDocuments:
    """Tests for loading all domain documents."""

    def test_loads_all_domains(self):
        """Test that all 7 domain files are loaded."""
        docs = load_all_documents()
        assert len(docs) == len(DOMAINS)

    def test_all_domains_represented(self):
        """Test each domain has at least one document."""
        docs = load_all_documents()
        domains_found = {doc.metadata["domain"] for doc in docs}
        for domain_key in DOMAINS:
            assert domain_key in domains_found, (
                f"Domain '{domain_key}' not found"
            )


class TestChunkDocuments:
    """Tests for document chunking."""

    def test_chunking_produces_output(self):
        """Test that chunking produces chunks."""
        docs = load_all_documents()
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > len(docs)

    def test_chunk_metadata_preserved(self):
        """Test that chunks inherit parent metadata."""
        docs = load_all_documents()
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "domain" in chunk.metadata
            assert "source" in chunk.metadata

    def test_chunk_size_respected(self):
        """Test chunks don't exceed max size (with some tolerance)."""
        docs = load_all_documents()
        chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=30)
        for chunk in chunks:
            # Allow some tolerance for splitting
            assert len(chunk.page_content) <= 600


class TestLoadAndChunkAll:
    """Tests for the convenience function."""

    def test_end_to_end(self):
        """Test loading and chunking in one step."""
        chunks = load_and_chunk_all()
        assert len(chunks) > 0
        assert all("domain" in c.metadata for c in chunks)
