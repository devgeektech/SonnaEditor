"""In-memory job registry with batched JSON persistence and restart recovery.

A `JobRecord` carries the full mutable state for one /api/process or /api/finetune
run: cumulative progress fields, the set of websocket subscribers, a cancellation
event, and a `threading.Lock` for sub-microsecond critical sections shared between
the API event-loop thread (readers) and the `asyncio.to_thread` worker (writer).

Persistence cadence (per the approved plan, not "every transition"):
- always on state transitions (queued → running → complete/cancelled/failed)
- every PERSIST_EVERY_N progress updates (currently 10) during running
- always on terminal state

On server boot, `recover_orphaned_jobs()` flips any persisted state == "running"
to "failed" with an explanatory error — a killed inference run is not safely
resumable. A pidfile prevents two server processes racing on the same dir.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


# ── Paths ──────────────────────────────────────────────────────────────────

JOBS_DIR = Path.home() / ".saha" / "jobs"
PIDFILE = JOBS_DIR / ".serve.pid"

# Persist every Nth photo / epoch update during a running job.
PERSIST_EVERY_N = 10

# Per-subscriber queue cap. Slow consumers get force-disconnected on overflow.
SUBSCRIBER_QUEUE_MAXSIZE = 100


# ── Types ──────────────────────────────────────────────────────────────────

JobState = Literal["queued", "running", "complete", "cancelled", "failed"]
JobKind = Literal["process", "finetune", "train"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class JobRecord:
    """Mutable state for one job. Always mutate under `self.lock`."""

    job_id: str
    kind: JobKind
    state: JobState = "queued"
    started_at: str = field(default_factory=_now_iso)
    ended_at: Optional[str] = None
    error: Optional[str] = None
    cancel_requested: bool = False

    # process-only
    folder_path: Optional[str] = None
    profile_id: Optional[str] = None
    photos_total: Optional[int] = None
    photos_processed: int = 0
    photos_flagged: int = 0
    photos_failed: int = 0
    current_photo: Optional[str] = None
    photos_per_sec: float = 0.0
    eta_seconds: int = 0
    output_paths_so_far: list[str] = field(default_factory=list)

    # finetune-only
    base_profile_id: Optional[str] = None
    captures_dir: Optional[str] = None
    profile_name: Optional[str] = None
    dataset_dir: Optional[str] = None
    epochs_total: Optional[int] = None
    epochs_completed: int = 0
    current_epoch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    new_checkpoint_path: Optional[str] = None

    # runtime-only (excluded from snapshot)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    loop: Optional[asyncio.AbstractEventLoop] = None
    _updates_since_persist: int = 0

    # Snapshot-only field for the UI: warns "this run is ~10× slower because
    # MC-dropout uncertainty is enabled". Set at job creation by the route.
    uncertainty_enabled: bool = False

    def snapshot_dict(self) -> dict[str, Any]:
        """Serialisable dict suitable for JSON persistence and API responses.

        Caller is responsible for holding `self.lock` if cross-thread coherence matters.
        """
        common = {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "uncertainty_enabled": self.uncertainty_enabled,
        }
        if self.kind == "process":
            common.update({
                "folder_path": self.folder_path,
                "profile_id": self.profile_id,
                "photos_total": self.photos_total,
                "photos_processed": self.photos_processed,
                "photos_flagged": self.photos_flagged,
                "photos_failed": self.photos_failed,
                "current_photo": self.current_photo,
                "photos_per_sec": self.photos_per_sec,
                "eta_seconds": self.eta_seconds,
                "output_paths_so_far": list(self.output_paths_so_far),
            })
        else:
            common.update({
                "folder_path": self.folder_path,
                "base_profile_id": self.base_profile_id,
                "captures_dir": self.captures_dir,
                "profile_name": self.profile_name,
                "dataset_dir": self.dataset_dir,
                "epochs_total": self.epochs_total,
                "epochs_completed": self.epochs_completed,
                "current_epoch": self.current_epoch,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "new_checkpoint_path": self.new_checkpoint_path,
            })
        return common


# ── Registry ───────────────────────────────────────────────────────────────

_registry: dict[str, JobRecord] = {}
_registry_lock = threading.Lock()


def create(kind: JobKind, **fields: Any) -> JobRecord:
    """Create a new JobRecord, register it, persist the initial snapshot."""
    job_id = str(uuid.uuid4())
    record = JobRecord(job_id=job_id, kind=kind, **fields)
    with _registry_lock:
        _registry[job_id] = record
    persist(record)
    return record


def get(job_id: str) -> Optional[JobRecord]:
    with _registry_lock:
        return _registry.get(job_id)


def all_records() -> list[JobRecord]:
    with _registry_lock:
        return list(_registry.values())


def transition(record: JobRecord, new_state: JobState, error: Optional[str] = None) -> None:
    """Move a job to a new state and persist immediately."""
    with record.lock:
        record.state = new_state
        if error is not None:
            record.error = error
        if new_state in ("complete", "cancelled", "failed"):
            record.ended_at = _now_iso()
    persist(record)


def note_progress(record: JobRecord) -> None:
    """Bump the per-update counter; persist if we've crossed PERSIST_EVERY_N.

    Caller MUST hold record.lock when mutating snapshot fields. This call may
    be made under-lock or after release; it acquires the lock briefly itself.
    """
    with record.lock:
        record._updates_since_persist += 1
        should_persist = record._updates_since_persist >= PERSIST_EVERY_N
        if should_persist:
            record._updates_since_persist = 0
    if should_persist:
        persist(record)


def cancel(record: JobRecord) -> None:
    with record.lock:
        record.cancel_requested = True
    record.cancel_event.set()
    persist(record)


def is_terminal(state: JobState) -> bool:
    return state in ("complete", "cancelled", "failed")


# ── Persistence ────────────────────────────────────────────────────────────

def persist(record: JobRecord) -> None:
    """Atomically write the record's snapshot to ~/.saha/jobs/<job_id>.json."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with record.lock:
        snapshot = record.snapshot_dict()
    target = JOBS_DIR / f"{record.job_id}.json"
    tmp = target.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(snapshot, indent=2))
        os.replace(tmp, target)
    except OSError as e:
        logger.warning("failed to persist job %s: %s", record.job_id, e)


# ── Recovery ───────────────────────────────────────────────────────────────

def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False

    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True

    try:
        os.kill(pid, 0)  # signal 0 = liveness probe
        return True
    except OSError:
        return False


def _check_pidfile() -> bool:
    """Return True if we can claim the pidfile (no other live serve process)."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    if PIDFILE.exists():
        try:
            old_pid = int(PIDFILE.read_text().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid is not None and old_pid != os.getpid():
            if _pid_exists(old_pid):
                logger.warning(
                    "another serve process (pid=%d) is already running; "
                    "skipping orphan recovery",
                    old_pid,
                )
                return False
    PIDFILE.write_text(str(os.getpid()))
    return True


def recover_orphaned_jobs() -> int:
    """Mark any persisted state == 'running' as failed. Returns count fixed."""
    if not _check_pidfile():
        return 0
    if not JOBS_DIR.is_dir():
        return 0

    fixed = 0
    for job_file in JOBS_DIR.glob("*.json"):
        try:
            data = json.loads(job_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("state") in ("queued", "running"):
            data["state"] = "failed"
            data["error"] = "Server restarted during job"
            data["ended_at"] = _now_iso()
            try:
                tmp = job_file.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, indent=2))
                os.replace(tmp, job_file)
                fixed += 1
            except OSError as e:
                logger.warning("failed to mark %s failed: %s", job_file, e)
    if fixed:
        logger.info("recovered %d orphaned job(s) on startup", fixed)
    return fixed


def reset_for_tests() -> None:
    """Wipe the in-memory registry. Tests only."""
    with _registry_lock:
        _registry.clear()
