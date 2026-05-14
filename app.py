"""
Senior Citizen Chatbot — Streamlit Application
An accessible, senior-friendly chatbot with RAG-powered domain assistance.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import APP_TITLE, APP_SUBTITLE, DOMAINS
from src.rag_pipeline import SeniorChatbotPipeline
from src.voice_handler import is_voice_available, text_to_speech
from src.utils import setup_logger, is_emergency_query
from mlflow_config import setup_mlflow, log_query

logger = setup_logger("streamlit_app")


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Senior Care Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS for Senior Accessibility
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Large, readable text */
    .stMarkdown p, .stMarkdown li {
        font-size: 18px !important;
        line-height: 1.7 !important;
    }

    /* Chat messages */
    .stChatMessage {
        font-size: 18px !important;
        padding: 16px !important;
        border-radius: 16px !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        font-size: 16px !important;
    }

    /* Large buttons */
    .stButton > button {
        font-size: 18px !important;
        padding: 12px 28px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    /* Emergency button */
    .emergency-btn button {
        background: linear-gradient(135deg, #ff4444, #cc0000) !important;
        color: white !important;
        font-size: 20px !important;
        padding: 16px 32px !important;
        border: none !important;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
    }

    /* Domain cards */
    .domain-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 16px;
        border-radius: 12px;
        color: white;
        margin: 8px 0;
        transition: transform 0.2s;
        cursor: pointer;
    }

    .domain-card:hover {
        transform: scale(1.02);
    }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 24px;
        color: white;
    }

    .main-header h1 {
        font-size: 42px !important;
        margin-bottom: 8px;
    }

    .main-header p {
        font-size: 20px !important;
        opacity: 0.9;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }

    .badge-ready { background: #28a745; color: white; }
    .badge-loading { background: #ffc107; color: #333; }

    /* Input field */
    .stTextInput input {
        font-size: 18px !important;
        padding: 14px !important;
        border-radius: 12px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = None
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False
if "initialized" not in st.session_state:
    st.session_state.initialized = False


# ──────────────────────────────────────────────
# Initialize Pipeline
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_pipeline():
    """Initialize the chatbot pipeline (cached)."""
    # Initialize MLflow
    try:
        setup_mlflow()
        logger.info("MLflow initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {e}")

    pipeline = SeniorChatbotPipeline()
    pipeline.initialize()
    return pipeline


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Settings")

    # Domain selection
    st.markdown("### 🎯 Choose a Topic")
    st.markdown(
        "*Select a topic to focus my answers, "
        "or leave on 'Auto-Detect'*"
    )

    domain_options = {"auto": "🔄 Auto-Detect (Recommended)"}
    for key, info in DOMAINS.items():
        domain_options[key] = f"{info['icon']} {info['label']}"

    selected = st.radio(
        "Domain",
        options=list(domain_options.keys()),
        format_func=lambda x: domain_options[x],
        index=0,
        label_visibility="collapsed",
    )

    if selected != "auto":
        st.session_state.selected_domain = selected
        info = DOMAINS[selected]
        st.info(f"**{info['label']}**: {info['description']}")
    else:
        st.session_state.selected_domain = None

    st.markdown("---")

    # Voice toggle
    voice_status = is_voice_available()
    st.markdown("### 🎙️ Voice Features")
    if voice_status["tts"]:
        st.session_state.voice_enabled = st.toggle(
            "🔊 Read responses aloud",
            value=st.session_state.voice_enabled,
        )
    else:
        st.warning("Voice features require pyttsx3. Install with: "
                    "`pip install pyttsx3`")

    st.markdown("---")

    # Emergency button
    st.markdown("### 🚨 Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚨 Emergency", use_container_width=True, type="primary"):
            st.session_state.messages.append({
                "role": "user",
                "content": "I need emergency help!",
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "🚨 **EMERGENCY CONTACTS:**\n\n"
                    "• **911** — Police, Fire, Ambulance\n"
                    "• **988** — Suicide & Crisis Lifeline\n"
                    "• **1-800-222-1222** — Poison Control\n\n"
                    "⚠️ If this is a medical emergency, "
                    "**call 911 immediately!**"
                ),
            })
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # About section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **Senior Care Assistant** is an AI-powered chatbot
        designed to help senior citizens with:

        - 🏥 Healthcare questions
        - 📱 Technology help
        - 🏦 Banking safety
        - 🚨 Emergency guidance
        - 🎭 Entertainment ideas
        - 💬 Friendly conversation
        - 🎙️ Voice & text tips

        *Your conversations are private and not stored
        beyond this session.*
        """)


# ──────────────────────────────────────────────
# Main Content Area
# ──────────────────────────────────────────────
# Header
st.markdown(f"""
<div class="main-header">
    <h1>{APP_TITLE}</h1>
    <p>{APP_SUBTITLE}</p>
</div>
""", unsafe_allow_html=True)

# Initialize pipeline
if not st.session_state.initialized:
    with st.spinner("🔄 Starting up... This may take a minute on first run."):
        try:
            st.session_state.pipeline = init_pipeline()
            st.session_state.initialized = True
            st.success("✅ Ready to help! Type your question below.")
        except Exception as e:
            st.error(f"❌ Error starting chatbot: {e}")
            logger.error(f"Pipeline init error: {e}")
            st.stop()

# Domain chips display
st.markdown("**💡 I can help you with:**")
cols = st.columns(7)
for i, (key, info) in enumerate(DOMAINS.items()):
    with cols[i]:
        st.markdown(
            f"<div style='text-align:center; padding:8px; "
            f"background:#f0f2f6; border-radius:10px; "
            f"font-size:14px;'>"
            f"{info['icon']}<br>{info['label']}</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "domain_badge" in msg:
            st.caption(msg["domain_badge"])

# Chat input
if prompt := st.chat_input(
    "Type your question here... (e.g., 'How do I video call my family?')"
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.pipeline.query(
                    user_query=prompt,
                    domain_override=st.session_state.selected_domain,
                )

                response = result["response"]
                domain_info = result["domain_info"]
                
                # Create a friendlier, non-technical badge
                if result['confidence'] > 0.2:
                    badge = f"Topic: {domain_info.get('icon', '💬')} {domain_info.get('label', 'General')}"
                else:
                    badge = "Topic: 💬 General"

                st.markdown(response)
                st.caption(badge)

                # Store assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "domain_badge": badge,
                })

                # Log to MLflow
                try:
                    log_query(
                        query=prompt,
                        domain=result["domain"],
                        confidence=result["confidence"],
                        response_length=len(response)
                    )
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")

                # Voice output
                if st.session_state.voice_enabled:
                    text_to_speech(response)

            except Exception as e:
                error_msg = (
                    "I'm sorry, I had trouble with that question. "
                    "Please try again or rephrase."
                )
                st.error(error_msg)
                logger.error(f"Query error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

# Welcome message if empty chat
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:40px; color:#666;">
        <h3>👋 Hello! I'm your Senior Care Assistant.</h3>
        <p style="font-size:18px;">
            Ask me anything about health, technology, banking,
            emergencies, entertainment, or just chat!
        </p>
        <p style="font-size:16px; color:#999;">
            Try: "How do I make a video call?" or
            "What should I do if I fall?"
        </p>
    </div>
    """, unsafe_allow_html=True)
