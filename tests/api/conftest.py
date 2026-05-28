"""Fixtures for API route tests — isolates per-test ~/.saha state and CHECKPOINTS_DIR."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from sonna_editor import config
from sonna_editor.api import jobs as jobs_module
from sonna_editor.api import server as server_module
from sonna_editor.api.routes import captures as captures_route
from sonna_editor.api.routes import finetune as finetune_route
from sonna_editor.api.routes import folders as folders_route
from sonna_editor.api.routes import profiles as profiles_route


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Redirect every disk path the API touches into a per-test tmp dir."""
    checkpoints_dir = tmp_path / "checkpoints"
    captures_dir = tmp_path / "captures"
    saha_dir = tmp_path / "saha"
    jobs_dir = saha_dir / "jobs"
    finetune_runs = saha_dir / "finetune_runs"
    checkpoints_dir.mkdir()
    captures_dir.mkdir()
    saha_dir.mkdir()
    jobs_dir.mkdir()
    finetune_runs.mkdir()

    monkeypatch.setattr(config, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(config, "CAPTURES_DIR", captures_dir)
    monkeypatch.setattr(profiles_route, "_ACTIVE_PROFILE_FILE",
                        saha_dir / "active_profile.txt")
    monkeypatch.setattr(folders_route, "_RECENT_FOLDERS_FILE",
                        saha_dir / "recent_folders.json")
    monkeypatch.setattr(jobs_module, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(jobs_module, "PIDFILE", jobs_dir / ".serve.pid")
    monkeypatch.setattr(finetune_route, "_FINETUNE_RUNS_DIR", finetune_runs)

    # Wipe the in-memory job registry between tests.
    jobs_module.reset_for_tests()

    return {
        "checkpoints_dir": checkpoints_dir,
        "captures_dir": captures_dir,
        "saha_dir": saha_dir,
        "jobs_dir": jobs_dir,
        "finetune_runs": finetune_runs,
    }


@pytest.fixture
def client(isolated_paths: dict[str, Path]) -> Iterator[TestClient]:
    # Use the context-manager form so the lifespan / underlying anyio portal
    # persists across requests — required for asyncio.create_task background
    # jobs to outlive the request that spawned them.
    with TestClient(server_module.app) as c:
        yield c
