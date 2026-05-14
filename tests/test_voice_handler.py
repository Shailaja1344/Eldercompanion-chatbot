"""Tests for the voice handler module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from src.voice_handler import is_voice_available


class TestVoiceAvailability:
    """Tests for voice feature availability checking."""

    def test_returns_dict(self):
        result = is_voice_available()
        assert isinstance(result, dict)
        assert "tts" in result
        assert "stt" in result

    def test_values_are_boolean(self):
        result = is_voice_available()
        assert isinstance(result["tts"], bool)
        assert isinstance(result["stt"], bool)


class TestTextToSpeech:
    """Tests for TTS functionality (mocked)."""

    @patch("src.voice_handler._TTS_AVAILABLE", False)
    def test_tts_unavailable(self):
        from src.voice_handler import text_to_speech
        result = text_to_speech("Hello")
        assert result is False

    @patch("src.voice_handler._TTS_AVAILABLE", True)
    @patch("src.voice_handler.pyttsx3")
    def test_tts_success(self, mock_pyttsx3):
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = [
            MagicMock(id="voice1"),
            MagicMock(id="voice2"),
        ]
        from src.voice_handler import text_to_speech
        result = text_to_speech("Hello senior friend")
        assert result is True


class TestSpeechToText:
    """Tests for STT functionality (mocked)."""

    @patch("src.voice_handler._STT_AVAILABLE", False)
    def test_stt_unavailable(self):
        from src.voice_handler import speech_to_text
        result = speech_to_text()
        assert result == ""
