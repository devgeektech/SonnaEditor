"""Tests for WS /api/jobs/{id}/stream."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from .test_process_route import _fake_inference_factory, _make_ckpt, _make_raws


def test_websocket_emits_photo_complete_and_terminal(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=3)

    fake = _fake_inference_factory(3, sleep_per=0.02)
    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=fake):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
        }).json()
        job_id = ack["job_id"]

        with client.websocket_connect(f"/api/jobs/{job_id}/stream") as ws:
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] in ("job_complete", "job_failed", "job_cancelled"):
                    break

    photo_msgs = [m for m in messages if m["type"] == "photo_complete"]
    snapshot_msgs = [m for m in messages if m["type"] == "job_snapshot"]
    terminal_msgs = [m for m in messages if m["type"] == "job_complete"]
    assert len(snapshot_msgs) == 1
    assert snapshot_msgs[0]["photos_total"] == 3
    assert len(photo_msgs) == 3
    assert len(terminal_msgs) == 1

    first = photo_msgs[0]
    assert "edit_summary" in first
    assert "·" in first["edit_summary"]
    assert "Exp " in first["edit_summary"]
    assert "WB " in first["edit_summary"]
    assert first["status"] == "ok"
    assert first["photos_total"] == 3


def test_websocket_unknown_job_returns_error_message(client: TestClient) -> None:
    with client.websocket_connect("/api/jobs/no-such-id/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_websocket_terminal_replay_after_completion(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    """Connecting after a job completes should still get one terminal message."""
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=1)

    fake = _fake_inference_factory(1)
    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=fake):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
        }).json()
        job_id = ack["job_id"]
        # Wait for terminal
        for _ in range(50):
            if client.get(f"/api/jobs/{job_id}").json()["state"] == "complete":
                break
            time.sleep(0.05)

    with client.websocket_connect(f"/api/jobs/{job_id}/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "job_complete"
