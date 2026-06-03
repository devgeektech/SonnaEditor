"""Tests for POST /api/finetune."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient


def _make_ckpt(dir_: Path, name: str) -> Path:
    p = dir_ / name
    p.write_bytes(b"fake")
    return p


def test_finetune_unknown_base_profile(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    resp = client.post("/api/finetune", json={
        "base_profile_id": "missing",
        "captures_dir": "/abs/path",
    })
    assert resp.status_code == 404


def test_finetune_missing_captures_parquet(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    captures_dir = tmp_path / "ft_captures"
    captures_dir.mkdir()
    resp = client.post("/api/finetune", json={
        "base_profile_id": "dp-event-v1.0.1",
        "captures_dir": str(captures_dir),
    })
    assert resp.status_code == 400


def test_finetune_happy_path_with_mocks(
    client: TestClient, isolated_paths: dict[str, Path], tmp_path: Path
) -> None:
    _make_ckpt(isolated_paths["checkpoints_dir"], "model-v1.0.1.ckpt")
    captures_dir = tmp_path / "ft_captures"
    captures_dir.mkdir()
    pd.DataFrame({"id": ["a"], "deltas": ["{}"]}).to_parquet(
        captures_dir / "captures.parquet", index=False
    )

    epochs_seen: list[int] = []

    def fake_finetune(*, on_epoch_complete=None, **kw):
        if on_epoch_complete is not None:
            for e in range(3):
                on_epoch_complete({"epoch": e, "train_loss": 0.001 - 0.0001 * e,
                                    "val_loss": 0.0011 - 0.0001 * e})
                epochs_seen.append(e)
        return {
            "checkpoint_path": str(kw["output_dir"] / "model-v1.0.2.ckpt"),
            "ft_val_loss": 0.0008, "improved": True, "checkpoint_status": "promoted",
        }

    def fake_prepare(**kw):
        kw["output_path"].write_bytes(b"x")
        return None

    with patch("sonna_editor.api.routes.finetune.finetune_retrain.finetune_model",
               side_effect=fake_finetune), \
         patch("sonna_editor.api.routes.finetune.finetune_delta.prepare_finetune_dataset",
               side_effect=fake_prepare), \
         patch("sonna_editor.api.routes.finetune.pd.read_parquet",
               return_value=pd.DataFrame({"id": ["a"]})):
        ack = client.post("/api/finetune", json={
            "base_profile_id": "dp-event-v1.0.1",
            "captures_dir": str(captures_dir),
        }).json()
        job_id = ack["job_id"]

        for _ in range(50):
            snap = client.get(f"/api/jobs/{job_id}").json()
            if snap["state"] in ("complete", "failed", "cancelled"):
                break
            time.sleep(0.05)

    assert snap["state"] == "complete"
    assert snap["kind"] == "finetune"
    assert snap["epochs_completed"] == 3
    assert snap["new_checkpoint_path"].endswith("model-v1.0.2.ckpt")
    assert epochs_seen == [0, 1, 2]
