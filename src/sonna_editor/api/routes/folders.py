"""Folder scan + recent-folders list."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from sonna_editor import config
from sonna_editor.api.models import (
    FolderScanRequest,
    FolderScanResponse,
    RawFileEntry,
    RecentFolder,
)
from sonna_editor.data.catalog import CatalogError, connect_catalog, find_edited_photos

router = APIRouter()

_MAX_FILES_RETURNED = 500
_RECENT_FOLDERS_FILE = config.RECENT_FOLDERS_PATH


def _scan_raws(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_RAW_EXTENSIONS
    )


def raw_paths_from_catalog(catalog_path: Path) -> list[Path]:
    """Return existing RAW paths referenced by a Lightroom catalog.

    The catalog is opened read-only via `connect_catalog`; missing files,
    rejected rows, duplicates, and unsupported image extensions are excluded.
    """
    conn = connect_catalog(catalog_path)
    try:
        photos = find_edited_photos(conn)
    finally:
        conn.close()

    raws: list[Path] = []
    seen: set[Path] = set()
    for photo in photos:
        if photo.get("is_missing"):
            continue
        raw = Path(photo["file_path"])
        if raw.suffix.lower() not in config.SUPPORTED_RAW_EXTENSIONS:
            continue
        try:
            resolved = raw.resolve()
        except OSError:
            resolved = raw
        if resolved in seen:
            continue
        if not raw.is_file():
            continue
        seen.add(resolved)
        raws.append(raw)
    return sorted(raws, key=lambda p: (str(p.parent).lower(), p.name.lower()))


def _scan_catalog(catalog_path: Path, raw_path_str: str) -> FolderScanResponse:
    if not catalog_path.exists():
        return FolderScanResponse(
            folder_path=raw_path_str,
            source_type="catalog",
            raw_count=0,
            files=[],
            is_valid=False,
            error="Catalog does not exist",
        )
    if not catalog_path.is_file():
        return FolderScanResponse(
            folder_path=raw_path_str,
            source_type="catalog",
            raw_count=0,
            files=[],
            is_valid=False,
            error="Path is not a catalog file",
        )
    if catalog_path.suffix.lower() != ".lrcat":
        return FolderScanResponse(
            folder_path=raw_path_str,
            source_type="catalog",
            raw_count=0,
            files=[],
            is_valid=False,
            error="Path is not a Lightroom catalog (.lrcat)",
        )

    try:
        raws = raw_paths_from_catalog(catalog_path)
    except CatalogError as exc:
        return FolderScanResponse(
            folder_path=raw_path_str,
            source_type="catalog",
            raw_count=0,
            files=[],
            is_valid=False,
            error=str(exc),
        )

    if not raws:
        return FolderScanResponse(
            folder_path=raw_path_str,
            source_type="catalog",
            raw_count=0,
            files=[],
            is_valid=False,
            error="No available RAW files found in this catalog",
        )

    truncated = len(raws) > _MAX_FILES_RETURNED
    visible = raws[:_MAX_FILES_RETURNED]
    xmp_conflict_count = sum(1 for raw in raws if raw.with_suffix(".xmp").exists())
    return FolderScanResponse(
        folder_path=raw_path_str,
        source_type="catalog",
        raw_count=len(raws),
        files=[RawFileEntry(name=p.name, size_bytes=p.stat().st_size) for p in visible],
        is_valid=True,
        error=None,
        truncated=truncated,
        xmp_conflict_count=xmp_conflict_count,
    )


@router.post("/folders/scan", response_model=FolderScanResponse)
def scan_folder(req: FolderScanRequest) -> FolderScanResponse:
    raw_path_str = req.folder_path
    folder = Path(raw_path_str)

    if not folder.is_absolute():
        raise HTTPException(status_code=400, detail="Path must be absolute")

    if req.source_type == "catalog" or folder.suffix.lower() == ".lrcat":
        return _scan_catalog(folder, raw_path_str)

    if not folder.exists():
        return FolderScanResponse(
            folder_path=raw_path_str,
            source_type="folder",
            raw_count=0,
            files=[],
            is_valid=False,
            error="Folder does not exist",
        )
    if not folder.is_dir():
        return FolderScanResponse(
            folder_path=raw_path_str,
            source_type="folder",
            raw_count=0,
            files=[],
            is_valid=False,
            error="Path is not a folder",
        )

    raws = _scan_raws(folder)
    if not raws:
        return FolderScanResponse(
            folder_path=raw_path_str,
            source_type="folder",
            raw_count=0,
            files=[],
            is_valid=False,
            error="No RAW files found in this folder",
        )

    truncated = len(raws) > _MAX_FILES_RETURNED
    visible = raws[:_MAX_FILES_RETURNED]
    # Count XMPs that would collide with a Saha write: same basename as one of
    # the RAWs. The pipeline writes raw_path.with_suffix(".xmp") in place, so
    # basename-match is the exact collision predicate. Stray .xmp files
    # unrelated to any RAW are not counted (they aren't on the write path).
    xmp_conflict_count = sum(
        1 for raw in raws if (raw.parent / f"{raw.stem}.xmp").exists()
    )
    return FolderScanResponse(
        folder_path=raw_path_str,
        source_type="folder",
        raw_count=len(raws),
        files=[RawFileEntry(name=p.name, size_bytes=p.stat().st_size) for p in visible],
        is_valid=True,
        error=None,
        truncated=truncated,
        xmp_conflict_count=xmp_conflict_count,
    )


def record_recent_folder(folder_path: str, raw_count: int) -> None:
    """Prepend a folder to the recent list, dedupe by path, cap at 10.

    Called from the /api/process route after a job is queued so the UI's
    Recent folders panel stays in sync.
    """
    name = Path(folder_path).name
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    new_entry = {
        "name": name,
        "path": folder_path,
        "raw_count": raw_count,
        "last_processed_at": now_iso,
    }

    existing: list[dict] = []
    if _RECENT_FOLDERS_FILE.exists():
        try:
            data = json.loads(_RECENT_FOLDERS_FILE.read_text())
            if isinstance(data, list):
                existing = [
                    e for e in data
                    if isinstance(e, dict) and e.get("path") != folder_path
                ]
        except (json.JSONDecodeError, OSError):
            existing = []

    combined = [new_entry] + existing
    _RECENT_FOLDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RECENT_FOLDERS_FILE.write_text(json.dumps(combined[:10], indent=2))


@router.get("/folders/recent", response_model=list[RecentFolder])
def recent_folders() -> list[RecentFolder]:
    if not _RECENT_FOLDERS_FILE.exists():
        return []
    try:
        data = json.loads(_RECENT_FOLDERS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []

    out: list[RecentFolder] = []
    for entry in data[:10]:
        try:
            out.append(RecentFolder(
                name=entry.get("name") or Path(entry["path"]).name,
                path=entry["path"],
                raw_count=int(entry.get("raw_count", 0)),
                last_processed_at=entry.get("last_processed_at"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out
