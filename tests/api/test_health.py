"""Tests for GET /api/health."""

from __future__ import annotations

import torch
from fastapi.testclient import TestClient


from sonna_editor import config


def test_health_returns_ok_with_device(client: TestClient) -> None:
    expected_device = "mps" if torch.backends.mps.is_available() else "cpu"
    expected_model_loaded = (
        config.CHECKPOINTS_DIR.is_dir()
        and any(config.CHECKPOINTS_DIR.glob("model-v*.ckpt"))
    )

    resp = client.get("/api/health")
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is expected_model_loaded
    assert payload["device"] == expected_device
    assert isinstance(payload["version"], str) and payload["version"]
