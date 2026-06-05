from __future__ import annotations

import json
from pathlib import Path

from sonna_editor import config
from sonna_editor.foundation import (
    describe_foundation_checkpoint,
    ensure_foundation_repo_layout,
    list_foundation_versions,
    promote_foundation_checkpoint,
    resolve_foundation_checkpoint,
    rollback_foundation_checkpoint,
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
    assert manifest["schema_version"] == 2
    assert manifest["active_version"] == "foundation-test"
    assert manifest["active_checkpoint"] == "checkpoints/foundation-test.ckpt"
    assert manifest["versions"][0]["version"] == "foundation-test"
    assert manifest["versions"][0]["sha256"] is not None


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
    assert manifest["active_version"] == "foundation-002"
    assert manifest["active_checkpoint"] == "checkpoints/foundation-002.ckpt"
    assert manifest["history"][-1]["checkpoint"] == "checkpoints/foundation-001.ckpt"

    promoted_second.unlink()

    assert resolve_foundation_checkpoint() == promoted_first


def test_foundation_auto_versions_and_explicit_rollback(
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
        source_run_dir=source_dir,
    )
    promoted_second = promote_foundation_checkpoint(
        source_ckpt=second,
        display_name="Foundation Second",
        source_run_dir=source_dir,
    )

    assert promoted_first.name == "foundation-v1.ckpt"
    assert promoted_second.name == "foundation-v2.ckpt"
    assert [v["version"] for v in list_foundation_versions()] == [
        "foundation-v1",
        "foundation-v2",
    ]

    active = rollback_foundation_checkpoint("foundation-v1")

    assert active == promoted_first
    assert resolve_foundation_checkpoint() == promoted_first
    manifest = json.loads((repo / "foundation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_version"] == "foundation-v1"
    assert manifest["active_checkpoint"] == "checkpoints/foundation-v1.ckpt"

    provenance = describe_foundation_checkpoint(promoted_first)
    assert provenance["foundation_version"] == "foundation-v1"
    assert provenance["foundation_checkpoint"] == str(promoted_first.resolve())
    assert provenance["foundation_sha256"] is not None


def test_legacy_manifest_versions_include_active_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "foundation"
    checkpoints = repo / "checkpoints"
    checkpoints.mkdir(parents=True)
    monkeypatch.setattr(config, "FOUNDATION_REPO_DIR", repo)
    monkeypatch.delenv(config.FOUNDATION_REPO_ENV_VAR, raising=False)

    (checkpoints / "foundation-old.ckpt").write_bytes(b"old")
    (checkpoints / "foundation-active.ckpt").write_bytes(b"active")
    (repo / "foundation_manifest.json").write_text(
        json.dumps(
            {
                "active_checkpoint": "checkpoints/foundation-active.ckpt",
                "active_sidecar": "checkpoints/foundation-active.json",
                "display_name": "Active Foundation",
                "source_run_dir": "data/training_workspace/foundation_runs/active",
                "updated_at": "2026-06-05T00:00:00Z",
                "foundation_type": "sonna_editor_slider_regression",
                "history": [
                    {
                        "checkpoint": "checkpoints/foundation-old.ckpt",
                        "sidecar": "checkpoints/foundation-old.json",
                        "display_name": "Old Foundation",
                        "source_run_dir": "data/training_workspace/foundation_runs/old",
                        "updated_at": "2026-06-04T00:00:00Z",
                        "foundation_type": "sonna_editor_slider_regression",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert [v["version"] for v in list_foundation_versions()] == [
        "foundation-old",
        "foundation-active",
    ]


def test_empty_manifest_reports_no_foundation_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "foundation"
    repo.mkdir()
    monkeypatch.setattr(config, "FOUNDATION_REPO_DIR", repo)
    monkeypatch.delenv(config.FOUNDATION_REPO_ENV_VAR, raising=False)
    monkeypatch.delenv(config.FOUNDATION_CHECKPOINT_ENV_VAR, raising=False)
    (repo / "foundation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "active_version": None,
                "active_checkpoint": None,
                "versions": [],
                "history": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        resolve_foundation_checkpoint()
    except FileNotFoundError as exc:
        assert "no active checkpoint" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for empty foundation manifest")
