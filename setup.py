"""Setup script for the Senior Citizen Chatbot package."""

from setuptools import setup, find_packages

setup(
    name="senior-citizen-chatbot",
    version="1.0.0",
    description="RAG-powered chatbot for senior citizens with LLM and multi-domain support",
    author="Senior Chatbot Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.2.0",
        "langchain-community>=0.2.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "faiss-cpu>=1.7.4",
        "streamlit>=1.30.0",
        "mlflow>=2.10.0",
    ],
    extras_require={
        "voice": ["pyttsx3>=2.90", "SpeechRecognition>=3.10.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "flake8>=6.1.0", "black>=23.10.0"],
    },
)
