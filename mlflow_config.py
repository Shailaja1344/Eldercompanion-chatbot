"""
MLflow configuration and experiment tracking utilities.
"""

import mlflow
from pathlib import Path

from src.config import MLFLOW_TRACKING_DIR

# Set tracking URI to local directory
TRACKING_URI = f"file://{MLFLOW_TRACKING_DIR.resolve()}"
EXPERIMENT_NAME = "senior-chatbot-rag"


def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_query(query: str, domain: str, confidence: float, response_length: int):
    """Log a single query interaction to MLflow."""
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "query": query[:200],
            "domain": domain,
        })
        mlflow.log_metrics({
            "confidence": confidence,
            "response_length": response_length,
        })


def log_index_build(num_documents: int, num_chunks: int, chunk_size: int):
    """Log index building metrics."""
    with mlflow.start_run(run_name="index_build"):
        mlflow.log_params({"chunk_size": chunk_size})
        mlflow.log_metrics({
            "num_documents": num_documents,
            "num_chunks": num_chunks,
        })
