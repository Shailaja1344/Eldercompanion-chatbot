"""
Voice handler: Speech-to-Text and Text-to-Speech for senior accessibility.
Voice features are optional — gracefully degrades if dependencies missing.
"""

from src.config import VOICE_RATE, VOICE_VOLUME, SPEECH_RECOGNITION_TIMEOUT
from src.utils import setup_logger

logger = setup_logger(__name__)

# Check optional dependencies
_TTS_AVAILABLE = False
_STT_AVAILABLE = False

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    logger.warning("pyttsx3 not installed — TTS disabled")

try:
    import speech_recognition as sr
    _STT_AVAILABLE = True
except ImportError:
    logger.warning("SpeechRecognition not installed — STT disabled")


def is_voice_available() -> dict:
    """Check which voice features are available."""
    return {
        "tts": _TTS_AVAILABLE,
        "stt": _STT_AVAILABLE,
    }


def text_to_speech(
    text: str,
    rate: int = VOICE_RATE,
    volume: float = VOICE_VOLUME,
) -> bool:
    """
    Convert text to speech using pyttsx3 (offline).
    Speaks at a slower rate suitable for seniors.
    """
    if not _TTS_AVAILABLE:
        logger.warning("TTS not available — pyttsx3 not installed")
        return False

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)

        # Try to use a clear, friendly voice
        voices = engine.getProperty("voices")
        if len(voices) > 1:
            engine.setProperty("voice", voices[1].id)

        engine.say(text)
        engine.runAndWait()
        engine.stop()
        logger.info(f"TTS completed ({len(text)} chars)")
        return True
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False


def speech_to_text(
    timeout: int = SPEECH_RECOGNITION_TIMEOUT,
) -> str:
    """
    Capture speech from microphone and convert to text.
    Uses Google Speech Recognition (requires internet).
    """
    if not _STT_AVAILABLE:
        logger.warning("STT not available — SpeechRecognition not installed")
        return ""

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as source:
            logger.info("Listening... (speak now)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=timeout)

        logger.info("Processing speech...")
        text = recognizer.recognize_google(audio)
        logger.info(f"STT result: '{text}'")
        return text

    except sr.WaitTimeoutError:
        logger.warning("No speech detected within timeout")
        return ""
    except sr.UnknownValueError:
        logger.warning("Could not understand audio")
        return ""
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        return ""
    except Exception as e:
        logger.error(f"STT error: {e}")
        return ""
