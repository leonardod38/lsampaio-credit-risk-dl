"""Utilitários compartilhados para configuração do MLflow."""

import os
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_MLRUNS = ROOT / "mlruns"


def can_connect_to_tracking(tracking_uri: str) -> bool:
    """Retorna True quando o endpoint do MLflow está acessível."""
    if not tracking_uri or tracking_uri.startswith("file:"):
        return False

    try:
        parsed = tracking_uri.replace("http://", "").replace("https://", "")
        host = parsed.split(":", 1)[0]
        port = int(parsed.rsplit(":", 1)[-1].split("/", 1)[0])
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Usa o servidor MLflow quando disponível e cai para armazenamento local."""
    candidate = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"
    if can_connect_to_tracking(candidate):
        return candidate

    LOCAL_MLRUNS.mkdir(exist_ok=True)
    local_uri = LOCAL_MLRUNS.resolve().as_posix()
    return f"file:///{local_uri}"


def configure_mlflow(experiment_name: str, tracking_uri: str | None = None):
    """Configura MLflow com fallback local quando o servidor não está ativo."""
    import mlflow

    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri
