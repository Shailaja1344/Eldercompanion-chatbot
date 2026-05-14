"""
Central configuration for the Senior Citizen Chatbot.
All paths, model settings, and domain definitions are managed here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = PROJECT_ROOT / "vectorstore"
LOGS_DIR = PROJECT_ROOT / "logs"
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"

# Ensure directories exist
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
# Embedding model (runs locally, ~80MB)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model (runs via Groq API)
LLM_MODEL_NAME = "llama-3.1-8b-instant"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.3

# ──────────────────────────────────────────────
# RAG Configuration
# ──────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 4
FAISS_INDEX_NAME = "senior_chatbot_index"

# ──────────────────────────────────────────────
# Domain Definitions
# ──────────────────────────────────────────────
DOMAINS = {
    "healthcare": {
        "file": "healthcare.txt",
        "icon": "🏥",
        "label": "Healthcare",
        "description": "Medication, nutrition, exercise, and doctor visits",
        "keywords": [
            "medicine", "medication", "doctor", "hospital", "health",
            "pain", "symptom", "blood pressure", "diabetes", "heart",
            "diet", "nutrition", "exercise", "vitamin", "pharmacy",
            "prescription", "checkup", "fever", "cough", "flu",
            "arthritis", "cholesterol", "sleep", "hydration", "calcium",
        ],
    },
    "technology": {
        "file": "technology.txt",
        "icon": "📱",
        "label": "Technology",
        "description": "Smartphones, video calls, and internet safety",
        "keywords": [
            "phone", "smartphone", "computer", "laptop", "tablet",
            "internet", "wifi", "email", "app", "download",
            "password", "video call", "zoom", "whatsapp", "facetime",
            "scam", "phishing", "virus", "update", "setting",
        ],
    },
    "banking": {
        "file": "banking.txt",
        "icon": "🏦",
        "label": "Banking",
        "description": "Safe banking, pensions, and fraud prevention",
        "keywords": [
            "bank", "money", "account", "savings", "pension",
            "social security", "atm", "credit", "debit", "card",
            "fraud", "scam", "check", "deposit", "withdraw",
            "transfer", "loan", "interest", "tax", "investment",
        ],
    },
    "emergency": {
        "file": "emergency.txt",
        "icon": "🚨",
        "label": "Emergency",
        "description": "911, falls, medical emergencies, and safety",
        "keywords": [
            "emergency", "911", "help", "fall", "accident",
            "ambulance", "fire", "police", "poison", "bleeding",
            "chest pain", "stroke", "unconscious", "choking", "burn",
            "power outage", "earthquake", "tornado", "hurricane", "flood",
        ],
    },
    "entertainment": {
        "file": "entertainment.txt",
        "icon": "🎭",
        "label": "Entertainment",
        "description": "Music, movies, games, hobbies, and fun activities",
        "keywords": [
            "movie", "music", "song", "book", "read",
            "game", "puzzle", "crossword", "sudoku", "hobby",
            "garden", "craft", "paint", "knit", "cook",
            "tv", "show", "netflix", "youtube", "podcast",
            "audiobook", "bird", "museum", "dance", "fun",
        ],
    },
    "companion": {
        "file": "companion.txt",
        "icon": "💬",
        "label": "Companion",
        "description": "Emotional support, loneliness tips, and social connection",
        "keywords": [
            "lonely", "loneliness", "sad", "depressed", "anxious",
            "friend", "family", "talk", "chat", "company",
            "pet", "dog", "cat", "volunteer", "community",
            "mental health", "wellness", "mood", "grateful", "meditation",
            "hello", "hi", "how are you", "good morning", "thank you",
        ],
    },
    "voice_text": {
        "file": "voice_text_tips.txt",
        "icon": "🎙️",
        "label": "Voice & Text",
        "description": "Voice assistants, dictation, and accessibility features",
        "keywords": [
            "voice", "speak", "siri", "alexa", "google assistant",
            "dictation", "speech", "microphone", "text size", "font",
            "zoom", "magnifier", "hearing", "caption", "subtitle",
            "accessibility", "screen reader", "dark mode", "keyboard", "emoji",
        ],
    },
}

# ──────────────────────────────────────────────
# Voice Configuration
# ──────────────────────────────────────────────
VOICE_RATE = 150          # Words per minute (slower for seniors)
VOICE_VOLUME = 1.0        # 0.0 to 1.0
SPEECH_RECOGNITION_TIMEOUT = 10  # seconds

# ──────────────────────────────────────────────
# Streamlit UI Configuration
# ──────────────────────────────────────────────
APP_TITLE = "🤖 Senior Care Assistant"
APP_SUBTITLE = "Your friendly AI helper — ask me anything!"
FONT_SIZE_LARGE = "20px"
FONT_SIZE_BODY = "18px"
FONT_SIZE_SMALL = "16px"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "chatbot.log"

# ──────────────────────────────────────────────
# Senior-friendly System Prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a kind, patient, and engaging assistant designed \
specifically for senior citizens. You are a conversational partner who uses NLP, \
LLMs, and RAG technology to provide high-quality, thoughtful assistance.

Follow these rules:
1. Use simple, clear language but provide detailed and helpful explanations.
2. Be warm, respectful, and highly encouraging.
3. Provide multi-paragraph responses that elaborate on ideas and suggestions.
4. If the question is about a medical emergency, always advise calling 911 first.
5. Do NOT be too brief. Offer several options and explain why they might be enjoyable.
6. If you are unsure about something, say so honestly and suggest consulting \
a professional (doctor, banker, family member).
7. Always end on a very positive, supportive, and friendly note.

Use the following context to answer the question. Use your general knowledge to \
expand on the context and provide a richer, more conversational experience.

Context:
{context}

Question: {question}

Helpful Answer:"""
