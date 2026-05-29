import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlflow_utils import resolve_tracking_uri


def test_resolve_tracking_uri_falls_back_to_local_when_server_unavailable(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setattr("mlflow_utils.can_connect_to_tracking", lambda uri: False)

    uri = resolve_tracking_uri()

    assert uri.startswith("file:")


def test_resolve_tracking_uri_keeps_explicit_uri_when_available(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.test:5000")
    monkeypatch.setattr("mlflow_utils.can_connect_to_tracking", lambda uri: True)

    uri = resolve_tracking_uri()

    assert uri == "http://example.test:5000"
