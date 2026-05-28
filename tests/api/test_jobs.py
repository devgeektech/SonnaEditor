"""Tests for the in-memory job registry + persistence + recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonna_editor.api import jobs


def test_create_persists_initial_snapshot(isolated_paths: dict[str, Path]) -> None:
    record = jobs.create(kind="process", folder_path="/tmp/x", profile_id="p")
    persisted = isolated_paths["jobs_dir"] / f"{record.job_id}.json"
    assert persisted.exists()
    data = json.loads(persisted.read_text())
    assert data["state"] == "queued"
    assert data["kind"] == "process"
    assert data["folder_path"] == "/tmp/x"


def test_transition_persists_and_sets_ended_at(isolated_paths: dict[str, Path]) -> None:
    record = jobs.create(kind="process")
    jobs.transition(record, "running")
    jobs.transition(record, "complete")

    data = json.loads((isolated_paths["jobs_dir"] / f"{record.job_id}.json").read_text())
    assert data["state"] == "complete"
    assert data["ended_at"] is not None


def test_note_progress_batches_persistence(isolated_paths: dict[str, Path]) -> None:
    record = jobs.create(kind="process", photos_total=20)
    persisted = isolated_paths["jobs_dir"] / f"{record.job_id}.json"

    # Initial create wrote photos_processed=0 already.
    initial_mtime = persisted.stat().st_mtime_ns

    # Bump in-memory progress; persistence should NOT happen until N=10.
    for _ in range(jobs.PERSIST_EVERY_N - 1):
        with record.lock:
            record.photos_processed += 1
        jobs.note_progress(record)

    assert persisted.stat().st_mtime_ns == initial_mtime

    # The Nth call should trigger a persist.
    with record.lock:
        record.photos_processed += 1
    jobs.note_progress(record)
    data = json.loads(persisted.read_text())
    assert data["photos_processed"] == jobs.PERSIST_EVERY_N


def test_cancel_sets_event_and_flag(isolated_paths: dict[str, Path]) -> None:
    record = jobs.create(kind="process")
    jobs.cancel(record)
    assert record.cancel_event.is_set()
    data = json.loads((isolated_paths["jobs_dir"] / f"{record.job_id}.json").read_text())
    assert data["cancel_requested"] is True


def test_recover_orphaned_jobs(isolated_paths: dict[str, Path]) -> None:
    jobs_dir: Path = isolated_paths["jobs_dir"]
    (jobs_dir / "abc.json").write_text(json.dumps({
        "job_id": "abc", "kind": "process", "state": "running",
        "started_at": "2026-05-01T00:00:00Z",
    }))
    (jobs_dir / "def.json").write_text(json.dumps({
        "job_id": "def", "kind": "process", "state": "complete",
        "started_at": "2026-05-01T00:00:00Z",
    }))

    fixed = jobs.recover_orphaned_jobs()
    assert fixed == 1

    abc = json.loads((jobs_dir / "abc.json").read_text())
    assert abc["state"] == "failed"
    assert abc["error"] == "Server restarted during job"

    defjson = json.loads((jobs_dir / "def.json").read_text())
    assert defjson["state"] == "complete"
