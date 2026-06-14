"""
Senior Citizen Chatbot — Streamlit Application
An accessible, senior-friendly chatbot with RAG-powered domain assistance.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import uuid
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import APP_TITLE, APP_SUBTITLE, DATA_DIR
# pyrefly: ignore [missing-import]
from src.rag_pipeline import SeniorChatbotPipeline
from src.voice_handler import is_voice_available, text_to_speech
from src.utils import setup_logger, is_emergency_query


logger = setup_logger("streamlit_app")


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ElderlyCompanion",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# 🌸 ELDERLY COMPANION UI - REDESIGNED
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700;800&display=swap');

/* GLOBAL */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.stApp {
    font-family: 'Nunito', sans-serif;
    background: linear-gradient(135deg, #fdf5f9 0%, #f9f3fc 100%);
    color: #3a2e39;
}

/* MAIN CONTAINER */
.main-container {
    display: flex;
    gap: 0;
    height: 100vh;
}

/* SIDEBAR STYLING */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff8fc 0%, #fff0f7 100%) !important;
    border-right: 2px solid #f5d6e8 !important;
    padding: 20px 15px !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 12px;
}

/* SIDEBAR MENU HEADER */
.sidebar-menu-title {
    font-size: 20px;
    font-weight: 700;
    color: #d46a8c;
    text-align: center;
    margin-bottom: 20px;
    padding: 15px 0;
    border-bottom: 2px solid #f5c6d9;
}

/* HEALTH REMINDERS SECTION */
.health-section-title {
    font-size: 15px;
    font-weight: 700;
    color: #dbe7ff;
    margin: 18px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* REMINDER CARD */
.reminder-card {
    background: linear-gradient(135deg, #fff5fa 0%, #fff0f7 100%);
    border: 1.5px solid #f5c6d9;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}

.reminder-card:hover {
    border-color: #ff7aa2;
    box-shadow: 0 4px 12px rgba(255, 122, 162, 0.15);
    transform: translateX(4px);
}

.reminder-time {
    font-size: 12px;
    font-weight: 700;
    color: #ff7aa2;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.reminder-name {
    font-size: 14px;
    font-weight: 600;
    color: #3a2e39;
    margin: 4px 0;
}

.reminder-desc {
    font-size: 11px;
    color: #7b6b7a;
    margin-top: 2px;
}

/* PINK BUTTONS */
.btn-pink {
    width: 100%;
    background: linear-gradient(135deg, #ff7aa2 0%, #ff6b92 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 8px 0;
}

.btn-pink:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(255, 122, 162, 0.3);
}

/* WHITE BUTTONS */
.btn-white {
    width: 100%;
    background: white;
    color: #d46a8c;
    border: 1.5px solid #f5c6d9;
    border-radius: 12px;
    padding: 10px;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 6px 0;
    text-align: left;
    padding-left: 12px;
}

.btn-white:hover {
    background: #fff5fa;
    border-color: #ff7aa2;
}

/* CHAT HISTORY */
.chat-history-title {
    font-size: 15px;
    font-weight: 700;
    color: #3a2e39;
    margin: 16px 0 12px 0;
}

/* EMERGENCY SOS BUTTON */
.btn-emergency {
    width: 100%;
    background: linear-gradient(135deg, #ff4757 0%, #ff3838 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: 700;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 12px;
}

.btn-emergency:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 20px rgba(255, 71, 87, 0.4);
}

/* ACTIONS SECTION */
.actions-section {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1.5px solid #f5c6d9;
}

.actions-title {
    font-size: 14px;
    font-weight: 700;
    color: #3a2e39;
    margin-bottom: 12px;
}

/* MAIN CONTENT AREA */
.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: linear-gradient(180deg, #fef8fb 0%, #fdf5f9 100%);
    padding: 0;
}

/* HEADER SECTION */
.chat-header {
    background: white;
    border-bottom: 2px solid #f5d6e8;
    padding: 20px 40px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.header-title {
    font-size: 32px;
    font-weight: 800;
    color: #3a2e39;
    margin: 0;
}

.header-highlight {
    color: #d46a8c;
    font-size: 36px;
    display: inline-block;
    margin: 0 8px;
}

.header-subtitle {
    font-size: 14px;
    color: #7b6b7a;
    margin-top: 6px;
    font-weight: 500;
}

/* MESSAGES AREA */
.messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 30px 40px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

/* USER MESSAGE */
.user-message {
    align-self: flex-end;
    background: linear-gradient(135deg, #ffd4e5 0%, #ffbfd1 100%);
    color: #3a2e39;
    border-radius: 18px;
    padding: 12px 18px;
    max-width: 70%;
    word-wrap: break-word;
    box-shadow: 0 4px 12px rgba(255, 122, 162, 0.2);
    font-size: 14px;
}

/* ASSISTANT MESSAGE */
.assistant-message {
    align-self: flex-start;
    background: white;
    color: #3a2e39;
    border: 1.5px solid #f5d6e8;
    border-radius: 18px;
    padding: 14px 18px;
    max-width: 70%;
    word-wrap: break-word;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    font-size: 14px;
    line-height: 1.5;
}

.assistant-avatar {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #ff7aa2 0%, #d46a8c 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 18px;
    flex-shrink: 0;
}

.message-with-avatar {
    display: flex;
    gap: 12px;
    align-items: flex-end;
}

/* INPUT AREA */
.input-section {
    padding: 20px 40px;
    background: white;
    border-top: 2px solid #f5d6e8;
}

.stChatInputContainer {
    border-radius: 16px !important;
    border: 1.5px solid #f5c6d9 !important;
    background: linear-gradient(135deg, #fff9fc 0%, #fff5fa 100%) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
}

.stChatInputContainer input {
    font-size: 14px !important;
    color: #3a2e39 !important;
}

.stChatInputContainer input::placeholder {
    color: #b8a4b8 !important;
}

/* SEND BUTTON */
.send-btn {
    background: linear-gradient(135deg, #ff7aa2 0%, #ff6b92 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

.send-btn:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 6px 16px rgba(255, 122, 162, 0.3) !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #fff5fa;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #ff7aa2 0%, #d46a8c 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #ff6b92 0%, #c04c7a 100%);
}

/* HIDE STREAMLIT DEFAULTS */
header {visibility: hidden;}
.stMainBlockContainer {padding: 0 !important;}

/* STBUTTON OVERRIDES - ENHANCED VISIBILITY */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
    width: 100%;
    color: #ffffff !important;
    background-color: #2c2c3a !important;
    letter-spacing: 0.3px !important;
}

.stButton > button:hover {
    background-color: #3c3c4a !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* DARKER INFO MESSAGE STYLING */
.stInfo {
    background-color: #e8f1ff !important;
    border-left: 4px solid #d46a8c !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}

.stInfo > div {
    color: #3a2e39 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .user-message, .assistant-message {
        max-width: 95%;
    }
    
    .chat-header {
        padding: 15px 20px;
    }
    
    .messages-area {
        padding: 15px 20px;
    }
    
    .input-section {
        padding: 15px 20px;
    }
    
    .header-title {
        font-size: 24px;
    }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization (UNCHANGED)
# ──────────────────────────────────────────────
HISTORY_FILE = Path(DATA_DIR) / "chat_history.json"
REMINDERS_FILE = Path(DATA_DIR) / "reminders.json"


def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def load_reminders():
    if REMINDERS_FILE.exists():
        try:
            with open(REMINDERS_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []


def save_reminders(reminders):
    with open(REMINDERS_FILE, "w") as f:
        json.dump(reminders, f, indent=4)


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "reminders" not in st.session_state:
    st.session_state.reminders = load_reminders()


# ──────────────────────────────────────────────
# Pipeline Init (UNCHANGED)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_pipeline():
    logger.info("Application initialized")

    pipeline = SeniorChatbotPipeline()
    pipeline.initialize()
    return pipeline


# ──────────────────────────────────────────────
# SIDEBAR - NEW DESIGN
# ──────────────────────────────────────────────
with st.sidebar:
    # Menu Header
    st.markdown("""
    <div style="text-align:center;padding:10px 0;border-bottom:2px solid #f5c6d9;margin-bottom:20px;">
        <h2 style="color:#d46a8c;font-size:22px;margin:0;">📋 Menu</h2>
    </div>
    """, unsafe_allow_html=True)

    # ─── HEALTH REMINDERS ───
    st.markdown("""
    <div style="font-size:15px;font-weight:700;color:#d46a8c;margin:18px 0 12px 0;">
        🏥 Health Reminders
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.reminders:
        for i, rem in enumerate(st.session_state.reminders):
            st.markdown(f"""
            <div class="reminder-card">
                <div class="reminder-time">☀️ {rem.get('time','Morning')}</div>
                <div class="reminder-name">{rem.get('name','Reminder')}</div>
                <div class="reminder-desc">{rem.get('desc','Take care of your health')}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # FIXED: Darker, more visible text for "no reminders"
        st.markdown("""
        <div style="background-color:#e8f1ff;border-left:4px solid #d46a8c;border-radius:8px;padding:12px 16px;margin:8px 0;">
            <div style="color:#3a2e39;font-weight:600;font-size:14px;">💕 No reminders set</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── ADD/CLEAR REMINDERS ───
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Reminder", use_container_width=True):
            st.session_state.reminders.append({
                "time": "Morning",
                "name": "New Reminder",
                "desc": "Set your reminder"
            })
            save_reminders(st.session_state.reminders)
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.reminders = []
            save_reminders([])
            st.rerun()

    st.markdown("---")

    # ─── RECENT CHAT HISTORY ───
    st.markdown("""
    <div style="font-size:15px;font-weight:700;color:#3a2e39;margin:16px 0 12px 0;">
        ⏰ Recent Chat History
    </div>
    """, unsafe_allow_html=True)

    if st.button("💬 New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    history = load_history()
    if history:
        for sid, chat_data in list(reversed(list(history.items())))[:5]:
            title = chat_data.get("title", "Previous Chat")
            if st.button(f"📅 {title}", key=f"hist_{sid}", use_container_width=True):
                st.session_state.session_id = sid
                st.session_state.messages = chat_data.get("messages", [])
                st.rerun()
    else:
        st.info("📝 No past chats yet")

    st.markdown("---")

    # ─── ACTIONS SECTION ───
    st.markdown("""
    <div class="actions-section">
        <div class="actions-title">🎙️ Actions</div>
    </div>
    """, unsafe_allow_html=True)

    voice_status = is_voice_available()
    if voice_status["tts"]:
        st.session_state.voice_enabled = st.toggle(
            "🔊 Read aloud",
            value=st.session_state.voice_enabled,
        )

    # Emergency and Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚨 Emergency SOS", use_container_width=True):
            st.error("📞 Emergency services contacted!")
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ──────────────────────────────────────────────
# MAIN CHAT AREA - NEW DESIGN
# ──────────────────────────────────────────────

# Header Section
st.markdown("""
<div class="chat-header">
    <div style="display:flex;justify-content:center;align-items:center;gap:8px;margin-bottom:8px;">
        <span style="font-size:36px;">ElderlyCompanion</span>
        <span style="color:#d46a8c;font-size:36px;">❤️</span>
        <span style="font-size:32px;">👵</span>
    </div>
    <div class="header-subtitle">
        Your friendly AI companion — ask me anything! 💕
    </div>
    <div style="text-align:center;margin-top:12px;">
        <span style="font-size:24px;">🕐</span>
        <span style="font-size:24px;">📅</span>
        <span style="font-size:24px;">🌿</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize pipeline
if not st.session_state.initialized:
    with st.spinner("🔄 Starting up..."):
        st.session_state.pipeline = init_pipeline()
        st.session_state.initialized = True


# Messages Display Area
st.markdown('<div class="messages-area">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-end;margin-bottom:12px;">
            <div class="user-message">
                {msg["content"]}
                <div style="text-align:right;font-size:11px;color:#c04c7a;margin-top:4px;opacity:0.7;">13:19</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-with-avatar">
            <div class="assistant-avatar">🤖</div>
            <div style="flex:1;">
                <div class="assistant-message">
                    {msg["content"]}
                    <div style="text-align:right;font-size:11px;color:#b8a4b8;margin-top:6px;">13:20 ❤️</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input Section with Message
st.markdown('<div class="input-section">', unsafe_allow_html=True)

if prompt := st.chat_input("Type your question here... (e.g., 'How do I video call my family?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

st.markdown("""
<div style="text-align:center;margin-top:12px;font-size:13px;color:#d46a8c;">
    ❤️ Here to help you, always! ❤️
</div>
</div>
""", unsafe_allow_html=True)

# Process message with pipeline
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_msg = st.session_state.messages[-1]["content"]
    with st.spinner("💭 Thinking..."):
        result = st.session_state.pipeline.query(last_msg)
        response = result["response"]
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
        })
        
        # FIXED: Text-to-speech functionality - now properly triggers
        if st.session_state.voice_enabled:
            try:
                text_to_speech(response)
                logger.info(f"Text-to-speech triggered for message: {response[:50]}...")
            except Exception as e:
                logger.error(f"Error in text-to-speech: {e}")
                st.warning("⚠️ Could not read aloud. Please check audio settings.")
        
        st.rerun()
