"""Tests for POST /api/process and GET/POST /api/jobs/{id}."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


def _make_ckpt(dir_: Path, name: str) -> Path:
    p = dir_ / name
    p.write_bytes(b"fake")
    return p


def _make_raws(folder: Path, n: int = 3) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"img_{i}.cr3").write_bytes(b"x")


def _fake_inference_factory(n_photos: int, sleep_per: float = 0.0):
    """Return a stand-in for process_shoot_with_model that fires the callback."""

    def fake(*, input_dir, model_path, on_photo_complete=None, cancel_event=None, **_):
        cancelled = False
        for i in range(n_photos):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            if sleep_per:
                time.sleep(sleep_per)
            if on_photo_complete is not None:
                on_photo_complete({
                    "name": f"img_{i}.cr3",
                    "raw_path": str(input_dir / f"img_{i}.cr3"),
                    "xmp_path": str(input_dir / f"img_{i}.xmp"),
                    "predicted_values": {
                        "Exposure2012": 0.1 * i,
                        "Temperature": 5000 + 10 * i,
                        "Shadows2012": 5 * i,
                    },
                    "std": None,
                    "status": "ok",
                    "elapsed_seconds": 0.01,
                })
        return {
            "processed": i + 1 if n_photos > 0 else 0,
            "failed": 0,
            "failures": [],
            "output_paths": [],
            "low_confidence": [],
            "predictions_path": None,
            "cancelled": cancelled,
        }
    return fake


def test_process_happy_path(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=3)

    fake = _fake_inference_factory(3)
    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=fake):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
        }).json()
        job_id = ack["job_id"]
        assert ack["state"] == "queued"

        # Poll until terminal
        for _ in range(50):
            snap = client.get(f"/api/jobs/{job_id}").json()
            if snap["state"] in ("complete", "failed", "cancelled"):
                break
            time.sleep(0.05)

    assert snap["state"] == "complete"
    assert snap["photos_processed"] == 3
    assert snap["photos_total"] == 3

    # recent_folders.json was updated
    recent = client.get("/api/folders/recent").json()
    assert any(r["path"] == str(folder) for r in recent)


def test_process_bad_folder(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    resp = client.post("/api/process", json={
        "folder_path": "/this/folder/does/not/exist",
        "profile_id": "dp-event-v1.0.1",
    })
    assert resp.status_code == 400


def test_process_bad_profile(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    folder = tmp_path / "shoot"
    _make_raws(folder)
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    resp = client.post("/api/process", json={
        "folder_path": str(folder),
        "profile_id": "does-not-exist",
    })
    assert resp.status_code == 404


def test_get_job_404(client: TestClient) -> None:
    resp = client.get("/api/jobs/no-such-id")
    assert resp.status_code == 404


def test_cancel_mid_run(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=20)

    # Slow-ish per-photo so we can race a cancel in mid-stream.
    fake = _fake_inference_factory(20, sleep_per=0.05)
    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=fake):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
        }).json()
        job_id = ack["job_id"]

        # Wait until at least one photo has been processed
        for _ in range(50):
            snap = client.get(f"/api/jobs/{job_id}").json()
            if snap["photos_processed"] >= 1:
                break
            time.sleep(0.02)

        cancel_resp = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["cancel_requested"] is True

        # Wait for terminal state
        for _ in range(100):
            snap = client.get(f"/api/jobs/{job_id}").json()
            if snap["state"] in ("complete", "failed", "cancelled"):
                break
            time.sleep(0.05)

    assert snap["state"] == "cancelled"
    assert 1 <= snap["photos_processed"] < 20


def test_cancel_on_completed_job_returns_409(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=2)

    fake = _fake_inference_factory(2)
    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=fake):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
        }).json()
        job_id = ack["job_id"]

        for _ in range(50):
            if client.get(f"/api/jobs/{job_id}").json()["state"] == "complete":
                break
            time.sleep(0.05)

    cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 409


def test_job_persists_to_disk(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=2)

    fake = _fake_inference_factory(2)
    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=fake):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
        }).json()
        job_id = ack["job_id"]
        for _ in range(50):
            if client.get(f"/api/jobs/{job_id}").json()["state"] == "complete":
                break
            time.sleep(0.05)

    persisted = isolated_paths["jobs_dir"] / f"{job_id}.json"
    assert persisted.exists()
    data = json.loads(persisted.read_text())
    assert data["state"] == "complete"


def test_process_preserve_wb_default_false(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    """Without preserve_wb in the body, the pipeline is called with preserve_wb=False."""
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=1)

    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {
            "processed": 1, "failed": 0, "failures": [],
            "output_paths": [], "low_confidence": [],
            "predictions_path": None, "cancelled": False,
        }

    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=capture):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
        }).json()
        for _ in range(50):
            if client.get(f"/api/jobs/{ack['job_id']}").json()["state"] == "complete":
                break
            time.sleep(0.05)

    assert captured.get("preserve_wb") is False


def test_process_preserve_wb_true_forwarded(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    """preserve_wb=true in the request body reaches process_shoot_with_model."""
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=1)

    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {
            "processed": 1, "failed": 0, "failures": [],
            "output_paths": [], "low_confidence": [],
            "predictions_path": None, "cancelled": False,
        }

    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=capture):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
            "preserve_wb": True,
        }).json()
        for _ in range(50):
            if client.get(f"/api/jobs/{ack['job_id']}").json()["state"] == "complete":
                break
            time.sleep(0.05)

    assert captured.get("preserve_wb") is True


def test_process_skip_fields_forwarded(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    """skip_fields in the request body reaches process_shoot_with_model as extra_skip_fields."""
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=1)

    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {
            "processed": 1, "failed": 0, "failures": [],
            "output_paths": [], "low_confidence": [],
            "predictions_path": None, "cancelled": False,
        }

    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=capture):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
            "skip_fields": ["Tint"],
        }).json()
        for _ in range(50):
            if client.get(f"/api/jobs/{ack['job_id']}").json()["state"] == "complete":
                break
            time.sleep(0.05)

    extra_skip_fields = captured.get("extra_skip_fields")
    assert isinstance(extra_skip_fields, list)
    assert extra_skip_fields == ["Tint"]


def test_process_auto_straighten_forwarded(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    """auto_straighten in the request body reaches process_shoot_with_model."""
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    folder = tmp_path / "shoot"
    _make_raws(folder, n=1)

    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {
            "processed": 1, "failed": 0, "failures": [],
            "output_paths": [], "low_confidence": [],
            "predictions_path": None, "cancelled": False,
        }

    with patch("sonna_editor.api.routes.process.inference_pipeline.process_shoot_with_model",
               side_effect=capture):
        ack = client.post("/api/process", json={
            "folder_path": str(folder),
            "profile_id": "dp-event-v1.0.1",
            "auto_straighten": True,
        }).json()
        for _ in range(50):
            if client.get(f"/api/jobs/{ack['job_id']}").json()["state"] == "complete":
                break
            time.sleep(0.05)

    assert captured.get("auto_straighten") is True


def test_profile_endpoint_returns_default_skip_fields(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    """A profile sidecar with default_skip_fields is surfaced through the API."""
    ckpt_dir = isolated_paths["checkpoints_dir"]
    ckpt_path = _make_ckpt(ckpt_dir, "model-v1.2.0.ckpt")
    sidecar_path = ckpt_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps({
        "display_name": "DP Event v1.2.0 (test)",
        "default_skip_fields": ["Tint"],
    }))

    resp = client.get("/api/profiles")
    profiles = resp.json()
    v120 = next((p for p in profiles if p["version"] == "v1.2.0"), None)
    assert v120 is not None, f"v1.2.0 profile not discovered: {profiles}"
    assert v120["default_skip_fields"] == ["Tint"]
