.PHONY: install lint format test run build index clean help

# ──────────────────────────────────────────────
# Senior Citizen Chatbot — Makefile
# ──────────────────────────────────────────────

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

lint: ## Run flake8 linter
	flake8 src/ tests/ app.py --config .flake8

format: ## Format code with black
	black src/ tests/ app.py --config pyproject.toml

format-check: ## Check code formatting without changes
	black --check src/ tests/ app.py --config pyproject.toml

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-quick: ## Run tests without coverage
	pytest tests/ -v

run: ## Run the Streamlit application
	streamlit run app.py --server.port 8501

index: ## Rebuild the FAISS vector index
	python -c "from src.data_loader import load_and_chunk_all; from src.embeddings import rebuild_index; rebuild_index(load_and_chunk_all())"

build: ## Build Docker image
	docker build -t senior-chatbot:latest .

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-stop: ## Stop Docker Compose
	docker-compose down

clean: ## Clean generated files
	rm -rf vectorstore/ logs/ mlruns/ htmlcov/ .coverage
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
