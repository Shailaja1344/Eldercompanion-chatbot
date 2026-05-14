# 🤖 Senior Citizen Chatbot — RAG-Powered AI Assistant

An end-to-end ML project with CI/CD pipeline, featuring a RAG (Retrieval-Augmented Generation) chatbot designed specifically for senior citizens. Provides domain-specific assistance across **7 domains** with voice and text interaction.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🏥 **Healthcare** | Medication management, nutrition, exercise guidance |
| 📱 **Technology** | Smartphone help, video calls, internet safety |
| 🏦 **Banking** | Safe banking, pensions, fraud prevention |
| 🚨 **Emergency** | 911 guidance, falls, medical emergencies |
| 🎭 **Entertainment** | Music, movies, games, hobbies |
| 💬 **Companion** | Emotional support, loneliness tips, social connection |
| 🎙️ **Voice & Text** | Voice assistants, dictation, accessibility features |

### Technical Highlights
- **LLM**: Google Flan-T5 (runs locally, no API key needed)
- **RAG**: LangChain + FAISS vector store + Sentence Transformers
- **Voice**: Speech-to-Text & Text-to-Speech support
- **CI/CD**: GitHub Actions pipeline (lint → test → build → deploy)
- **MLOps**: MLflow experiment tracking + DVC data versioning
- **Docker**: Containerized deployment with Docker Compose

---

## 📁 Project Structure

```
senior_chatbot/
├── .github/workflows/ci_cd.yml    # CI/CD pipeline
├── data/                          # 7 domain knowledge bases
├── src/                           # Core source code
│   ├── config.py                  # Central configuration
│   ├── data_loader.py             # Document loading & chunking
│   ├── embeddings.py              # FAISS vector store
│   ├── retriever.py               # Similarity search
│   ├── llm_engine.py              # LLM loading & inference
│   ├── domain_classifier.py       # Query domain routing
│   ├── rag_pipeline.py            # End-to-end RAG orchestration
│   ├── voice_handler.py           # STT & TTS
│   └── utils.py                   # Shared utilities
├── tests/                         # Unit tests
├── app.py                         # Streamlit application
├── Dockerfile                     # Container build
├── docker-compose.yml             # Container orchestration
├── Makefile                       # Dev workflow automation
├── requirements.txt               # Python dependencies
└── mlflow_config.py               # Experiment tracking
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
cd senior_chatbot
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Build the Vector Index
```bash
python -c "from src.data_loader import load_and_chunk_all; from src.embeddings import rebuild_index; rebuild_index(load_and_chunk_all())"
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Run Tests
```bash
pytest tests/ -v --cov=src
```

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:8501
```

---

## 🔄 CI/CD Pipeline

```
Git Push → Lint (flake8+black) → Test (pytest+coverage) → Build Docker → Push to GHCR → Deploy
```

The pipeline is defined in `.github/workflows/ci_cd.yml` and triggers on pushes to `main` and PRs.

---

## 📊 MLOps

- **MLflow**: Tracks queries, domain classification accuracy, and response quality
- **DVC**: Version control for knowledge base data files

---

## 🛠️ Development

```bash
# Format code
black src/ tests/ app.py

# Lint
flake8 src/ tests/ app.py

# Run tests
pytest tests/ -v --cov=src --cov-report=html
```

---

## 📄 License

This project is for educational and research purposes.
