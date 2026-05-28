"""Tests for /api/folders/scan and /api/folders/recent."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def _make_raws(folder: Path, n: int, ext: str = ".CR3") -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        p = folder / f"IMG_{i:05d}{ext}"
        p.write_bytes(b"x" * (1000 + i))
        files.append(p)
    return files


def test_scan_happy_path(client: TestClient, tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    _make_raws(folder, 3)

    resp = client.post("/api/folders/scan", json={"folder_path": str(folder)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is True
    assert data["raw_count"] == 3
    assert data["truncated"] is False
    assert data["error"] is None
    assert [f["name"] for f in data["files"]] == sorted(f["name"] for f in data["files"])
    assert all("size_bytes" in f for f in data["files"])


def test_scan_rejects_relative_path(client: TestClient) -> None:
    resp = client.post("/api/folders/scan", json={"folder_path": "relative/path"})
    assert resp.status_code == 400


def test_scan_truncated_past_500(client: TestClient, tmp_path: Path) -> None:
    folder = tmp_path / "huge"
    _make_raws(folder, 503)

    resp = client.post("/api/folders/scan", json={"folder_path": str(folder)})
    data = resp.json()
    assert data["is_valid"] is True
    assert data["raw_count"] == 503
    assert data["truncated"] is True
    assert len(data["files"]) == 500


def test_scan_missing_folder(client: TestClient, tmp_path: Path) -> None:
    resp = client.post("/api/folders/scan",
                       json={"folder_path": str(tmp_path / "nope")})
    data = resp.json()
    assert data["is_valid"] is False
    assert data["error"] == "Folder does not exist"


def test_scan_xmp_conflict_count_zero_when_no_xmps(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "shoot"
    _make_raws(folder, 3)

    resp = client.post("/api/folders/scan", json={"folder_path": str(folder)})
    assert resp.json()["xmp_conflict_count"] == 0


def test_scan_xmp_conflict_counts_basename_matches(
    client: TestClient, tmp_path: Path
) -> None:
    """Only .xmp sidecars whose basename matches a discovered RAW count."""
    folder = tmp_path / "shoot"
    raws = _make_raws(folder, 4)
    # Conflicts: 2 of 4 RAWs have sidecars.
    (folder / f"{raws[0].stem}.xmp").write_text("<x:xmpmeta/>")
    (folder / f"{raws[2].stem}.xmp").write_text("<x:xmpmeta/>")
    # Stray .xmp unrelated to any RAW basename — must NOT count.
    (folder / "notes.xmp").write_text("<x:xmpmeta/>")

    resp = client.post("/api/folders/scan", json={"folder_path": str(folder)})
    assert resp.json()["xmp_conflict_count"] == 2


def test_scan_xmp_conflict_count_spans_full_listing(
    client: TestClient, tmp_path: Path
) -> None:
    """The xmp count covers all RAWs even when the files list is truncated."""
    folder = tmp_path / "huge"
    raws = _make_raws(folder, 503)
    # Two sidecars beyond the 500-file truncation cutoff.
    (folder / f"{raws[501].stem}.xmp").write_text("<x:xmpmeta/>")
    (folder / f"{raws[502].stem}.xmp").write_text("<x:xmpmeta/>")

    resp = client.post("/api/folders/scan", json={"folder_path": str(folder)})
    data = resp.json()
    assert data["truncated"] is True
    assert data["xmp_conflict_count"] == 2


def test_scan_path_is_file(client: TestClient, tmp_path: Path) -> None:
    f = tmp_path / "not_a_folder.txt"
    f.write_text("hi")
    resp = client.post("/api/folders/scan", json={"folder_path": str(f)})
    data = resp.json()
    assert data["is_valid"] is False
    assert data["error"] == "Path is not a folder"


def test_scan_no_raws(client: TestClient, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("nothing here")

    resp = client.post("/api/folders/scan", json={"folder_path": str(empty)})
    data = resp.json()
    assert data["is_valid"] is False
    assert data["error"] == "No RAW files found in this folder"


def test_recent_returns_empty_when_file_missing(client: TestClient) -> None:
    resp = client.get("/api/folders/recent")
    assert resp.status_code == 200
    assert resp.json() == []


def test_recent_returns_entries_when_file_present(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    file = isolated_paths["saha_dir"] / "recent_folders.json"
    file.write_text(json.dumps([
        {"name": "JCDecaux Brisbane",
         "path": "/Volumes/X/JCDecaux Brisbane",
         "raw_count": 412,
         "last_processed_at": "2026-05-08T10:00:00Z"},
    ]))

    payload = client.get("/api/folders/recent").json()
    assert len(payload) == 1
    assert payload[0]["name"] == "JCDecaux Brisbane"
    assert payload[0]["raw_count"] == 412
