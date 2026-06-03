from __future__ import annotations

import json
from pathlib import Path

from sonna_editor import config
from sonna_editor.foundation import (
    ensure_foundation_repo_layout,
    promote_foundation_checkpoint,
    resolve_foundation_checkpoint,
)


def test_foundation_layout_and_resolution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "foundation"
    monkeypatch.setattr(config, "FOUNDATION_REPO_DIR", repo)
    monkeypatch.delenv(config.FOUNDATION_REPO_ENV_VAR, raising=False)
    monkeypatch.delenv(config.FOUNDATION_CHECKPOINT_ENV_VAR, raising=False)

    created = ensure_foundation_repo_layout()

    assert created == repo
    assert (repo / "README.md").exists()
    assert (repo / ".gitattributes").read_text(encoding="utf-8").startswith("*.ckpt")
    assert (repo / "checkpoints").is_dir()

    source_dir = tmp_path / "run"
    source_dir.mkdir()
    source_ckpt = source_dir / "model.ckpt"
    source_ckpt.write_bytes(b"fake-model")
    source_sidecar = source_dir / "model.json"
    source_sidecar.write_text('{"display_name": "Foundation"}', encoding="utf-8")

    promoted = promote_foundation_checkpoint(
        source_ckpt=source_ckpt,
        display_name="Foundation",
        version_stem="foundation-test",
        source_run_dir=source_dir,
    )

    assert promoted == repo / "checkpoints" / "foundation-test.ckpt"
    assert promoted.read_bytes() == b"fake-model"
    assert promoted.with_suffix(".json").exists()
    assert resolve_foundation_checkpoint() == promoted

    manifest = json.loads((repo / "foundation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_checkpoint"] == "checkpoints/foundation-test.ckpt"


def test_foundation_promotion_keeps_history_and_falls_back_when_active_removed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "foundation"
    monkeypatch.setattr(config, "FOUNDATION_REPO_DIR", repo)
    monkeypatch.delenv(config.FOUNDATION_REPO_ENV_VAR, raising=False)
    monkeypatch.delenv(config.FOUNDATION_CHECKPOINT_ENV_VAR, raising=False)

    source_dir = tmp_path / "run"
    source_dir.mkdir()
    first = source_dir / "first.ckpt"
    first.write_bytes(b"first")
    second = source_dir / "second.ckpt"
    second.write_bytes(b"second")

    promoted_first = promote_foundation_checkpoint(
        source_ckpt=first,
        display_name="Foundation First",
        version_stem="foundation-001",
        source_run_dir=source_dir,
    )
    promoted_second = promote_foundation_checkpoint(
        source_ckpt=second,
        display_name="Foundation Second",
        version_stem="foundation-002",
        source_run_dir=source_dir,
    )

    assert resolve_foundation_checkpoint() == promoted_second
    manifest = json.loads((repo / "foundation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_checkpoint"] == "checkpoints/foundation-002.ckpt"
    assert manifest["history"][-1]["checkpoint"] == "checkpoints/foundation-001.ckpt"

    promoted_second.unlink()

    assert resolve_foundation_checkpoint() == promoted_first
