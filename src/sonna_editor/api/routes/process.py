"""POST /api/process + GET/POST /api/jobs/{id} + WS /api/jobs/{id}/stream.

The route validates inputs, creates a JobRecord, and spawns the existing sync
``process_shoot_with_model`` in a worker thread via asyncio.to_thread. The
per-photo callback bridge lives in api.callbacks; the JSON-persisted registry
in api.jobs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from sonna_editor import config
from sonna_editor.api import callbacks, jobs
from sonna_editor.api.models import JobAck, JobSnapshot, ProcessRequest
from sonna_editor.api.routes import folders as folders_route
from sonna_editor.api.routes import profiles as profiles_route
from sonna_editor.inference import pipeline as inference_pipeline

router = APIRouter()
_logger = logging.getLogger(__name__)


def _resolve_profile_path(profile_id: str) -> Optional[Path]:
    for p in profiles_route._discover_profiles():
        if p.id == profile_id:
            return Path(p.checkpoint_path)
    return None


def _validate_process_source(req: ProcessRequest) -> tuple[Path, list[Path], str]:
    """Return (source_path, raws, source_type). Raise HTTPException on invalid input."""
    source_path = Path(req.folder_path)
    if not source_path.is_absolute():
        raise HTTPException(status_code=400, detail="Path must be absolute")

    if req.source_type == "catalog" or source_path.suffix.lower() == ".lrcat":
        if not source_path.exists():
            raise HTTPException(status_code=400, detail="Catalog does not exist")
        if not source_path.is_file() or source_path.suffix.lower() != ".lrcat":
            raise HTTPException(
                status_code=400,
                detail="Path is not a Lightroom catalog (.lrcat)",
            )
        try:
            raws = folders_route.raw_paths_from_catalog(source_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not raws:
            raise HTTPException(
                status_code=400,
                detail="No available RAW files found in this catalog",
            )
        return source_path, raws, "catalog"

    if not source_path.exists():
        raise HTTPException(status_code=400, detail="Folder does not exist")
    if not source_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a folder")
    raws = sorted(
        p for p in source_path.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_RAW_EXTENSIONS
    )
    if not raws:
        raise HTTPException(status_code=400,
                            detail="No RAW files found in this folder")
    return source_path, raws, "folder"


async def _run_process_job(
    record: jobs.JobRecord,
    source_path: Path,
    source_type: str,
    raw_paths: list[Path],
    model_path: Path,
    flag_low_confidence: bool,
    write_xmp_in_place: bool,
    preserve_wb: bool,
    skip_fields: list[str],
    auto_straighten: bool,
) -> None:
    """Background task: run the inference pipeline and emit terminal message."""
    jobs.transition(record, "running")
    started_at = time.monotonic()
    prepared_cb = callbacks.make_photo_prepared_callback(record, started_at)
    photo_cb = callbacks.make_photo_callback(record, started_at)

    input_dir = source_path if source_type == "folder" else source_path.parent
    output_dir = None if write_xmp_in_place or source_type == "catalog" else source_path

    try:
        result = await asyncio.to_thread(
            inference_pipeline.process_shoot_with_model,
            input_dir=input_dir,
            model_path=model_path,
            output_dir=output_dir,
            raw_paths=raw_paths if source_type == "catalog" else None,
            uncertainty=flag_low_confidence,
            preserve_wb=preserve_wb,
            extra_skip_fields=skip_fields,
            auto_straighten=auto_straighten,
            on_photo_prepared=prepared_cb,
            on_photo_complete=photo_cb,
            cancel_event=record.cancel_event,
        )
    except Exception as e:  # noqa: BLE001 — surface failures to the API
        _logger.exception("process job %s failed", record.job_id)
        jobs.transition(record, "failed", error=str(e))
        callbacks.broadcast_terminal(record, "job_failed")
        return

    with record.lock:
        record.photos_failed = result.get("failed", 0)
        cancelled = bool(result.get("cancelled", False))

    if cancelled:
        jobs.transition(record, "cancelled")
        callbacks.broadcast_terminal(record, "job_cancelled")
    else:
        jobs.transition(record, "complete")
        callbacks.broadcast_terminal(record, "job_complete")


@router.post("/process", response_model=JobAck)
async def start_process(req: ProcessRequest) -> JobAck:
    source_path, raws, source_type = _validate_process_source(req)

    model_path = _resolve_profile_path(req.profile_id)
    if model_path is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    record = jobs.create(
        kind="process",
        folder_path=str(source_path),
        profile_id=req.profile_id,
        photos_total=len(raws),
        uncertainty_enabled=req.flag_low_confidence,
    )
    record.loop = asyncio.get_running_loop()

    folders_route.record_recent_folder(str(source_path), raw_count=len(raws))

    asyncio.create_task(_run_process_job(
        record=record,
        source_path=source_path,
        source_type=source_type,
        raw_paths=raws,
        model_path=model_path,
        flag_low_confidence=req.flag_low_confidence,
        write_xmp_in_place=req.write_xmp_in_place,
        preserve_wb=req.preserve_wb,
        skip_fields=list(req.skip_fields),
        auto_straighten=req.auto_straighten,
    ))

    return JobAck(job_id=record.job_id, state=record.state)


@router.get("/jobs/{job_id}", response_model=JobSnapshot)
def get_job(job_id: str) -> JobSnapshot:
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    with record.lock:
        snap = record.snapshot_dict()
    return JobSnapshot(**snap)


@router.post("/jobs/{job_id}/cancel", response_model=JobSnapshot)
def cancel_job(job_id: str) -> JobSnapshot:
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    with record.lock:
        current = record.state
    if jobs.is_terminal(current):
        raise HTTPException(status_code=409,
                            detail=f"Job already in terminal state: {current}")
    jobs.cancel(record)
    with record.lock:
        snap = record.snapshot_dict()
    return JobSnapshot(**snap)


@router.websocket("/jobs/{job_id}/stream")
async def stream_job(ws: WebSocket, job_id: str) -> None:
    """Live websocket stream with an initial snapshot backfill."""
    await ws.accept()
    record = jobs.get(job_id)
    if record is None:
        await ws.send_json({"type": "error", "error": "Job not found"})
        await ws.close()
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=jobs.SUBSCRIBER_QUEUE_MAXSIZE)
    with record.lock:
        record.subscribers.append(queue)
        terminal_now = jobs.is_terminal(record.state)
        current_snap = record.snapshot_dict()
        terminal_snap = current_snap if terminal_now else None

    # If the job has already finished, deliver one terminal message and close.
    if terminal_now and terminal_snap is not None:
        msg_type = {
            "complete":  "job_complete",
            "cancelled": "job_cancelled",
            "failed":    "job_failed",
        }.get(terminal_snap["state"], "job_complete")
        try:
            await ws.send_json({"type": msg_type, **terminal_snap})
        finally:
            with record.lock:
                if queue in record.subscribers:
                    record.subscribers.remove(queue)
            await ws.close()
            return

    await ws.send_json({"type": "job_snapshot", **current_snap})

    try:
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
            if msg.get("type") in ("job_complete", "job_cancelled", "job_failed"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        _logger.warning("ws stream %s ended with %r", job_id, e)
    finally:
        with record.lock:
            if queue in record.subscribers:
                record.subscribers.remove(queue)
        try:
            await ws.close()
        except RuntimeError:
            pass  # already closed
