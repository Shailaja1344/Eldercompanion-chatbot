"""Tests for the domain classifier module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.domain_classifier import (
    classify_domain_keywords,
    classify_domain,
    get_domain_info,
)


class TestKeywordClassification:
    """Tests for keyword-based domain classification."""

    def test_healthcare_query(self):
        domain, conf = classify_domain_keywords(
            "What medicine should I take for blood pressure?"
        )
        assert domain == "healthcare"
        assert conf > 0

    def test_technology_query(self):
        domain, conf = classify_domain_keywords(
            "How do I make a video call on WhatsApp?"
        )
        assert domain == "technology"

    def test_banking_query(self):
        domain, conf = classify_domain_keywords(
            "How do I check my bank account for fraud?"
        )
        assert domain == "banking"

    def test_emergency_query(self):
        domain, conf = classify_domain_keywords(
            "I fell and I'm bleeding, should I call 911?"
        )
        assert domain == "emergency"

    def test_entertainment_query(self):
        domain, conf = classify_domain_keywords(
            "What are some fun puzzle games I can play?"
        )
        assert domain == "entertainment"

    def test_companion_query(self):
        domain, conf = classify_domain_keywords(
            "I feel lonely and sad today"
        )
        assert domain == "companion"

    def test_voice_text_query(self):
        domain, conf = classify_domain_keywords(
            "How do I use Siri voice assistant?"
        )
        assert domain == "voice_text"

    def test_unknown_defaults_to_companion(self):
        domain, conf = classify_domain_keywords(
            "xyzabc random gibberish"
        )
        assert domain == "companion"
        assert conf == 0.0


class TestClassifyDomain:
    """Tests for the main classify_domain function."""

    def test_returns_tuple(self):
        result = classify_domain("How do I take my medicine?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_confidence_range(self):
        _, conf = classify_domain("Tell me about health checkups")
        assert 0.0 <= conf <= 1.0


class TestGetDomainInfo:
    """Tests for domain info retrieval."""

    def test_known_domain(self):
        info = get_domain_info("healthcare")
        assert info["icon"] == "🏥"
        assert info["label"] == "Healthcare"

    def test_unknown_domain(self):
        info = get_domain_info("nonexistent")
        assert "icon" in info
        assert "label" in info
