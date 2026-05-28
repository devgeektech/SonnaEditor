"""POST /api/finetune — wraps finetune_model() as an async job.

Output checkpoint goes under ~/.saha/finetune_runs/<job_id>/. Promotion is
deliberately NOT automatic — after the job completes, the user must call
/api/profiles/{new_id}/activate to switch to it. This matches the Phase 5
"no auto-trigger" rule.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from sonna_editor import config
from sonna_editor.api import callbacks, jobs
from sonna_editor.api.models import FinetuneRequest, JobAck
from sonna_editor.api.routes import profiles as profiles_route
from sonna_editor.finetune import delta as finetune_delta
from sonna_editor.finetune import retrain as finetune_retrain

router = APIRouter()
_logger = logging.getLogger(__name__)

_FINETUNE_RUNS_DIR = Path.home() / ".saha" / "finetune_runs"


def _resolve_profile_path(profile_id: str) -> Optional[Path]:
    for p in profiles_route._discover_profiles():
        if p.id == profile_id:
            return Path(p.checkpoint_path)
    return None


def _validate_captures_dir(captures_dir_str: str) -> Path:
    captures_dir = Path(captures_dir_str)
    if not captures_dir.is_absolute():
        raise HTTPException(status_code=400, detail="Path must be absolute")
    captures_path = captures_dir / "captures.parquet"
    if not captures_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"captures.parquet not found in {captures_dir}",
        )
    return captures_dir


async def _run_finetune_job(
    record: jobs.JobRecord,
    base_checkpoint: Path,
    captures_dir: Path,
    weight_recent: float,
    max_epochs: int,
) -> None:
    jobs.transition(record, "running")

    run_dir = _FINETUNE_RUNS_DIR / record.job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    finetune_parquet = run_dir / "finetune.parquet"

    try:
        captures_df = pd.read_parquet(captures_dir / "captures.parquet")
        finetune_delta.prepare_finetune_dataset(
            captures=captures_df,
            original_train_parquet=config.ORIGINAL_TRAIN_PARQUET,
            output_path=finetune_parquet,
            weight_recent=weight_recent,
        )

        epoch_cb = callbacks.make_epoch_callback(record)

        # Output the new checkpoint into config.CHECKPOINTS_DIR so /api/profiles
        # discovers it without any change to the discovery code (Phase 6 will
        # replace this with a proper profile registry). The intermediate
        # finetune parquet stays under ~/.saha/finetune_runs/<job_id>/ since
        # it's job-scoped scratch data.
        result = await asyncio.to_thread(
            finetune_retrain.finetune_model,
            base_checkpoint=base_checkpoint,
            finetune_parquet=finetune_parquet,
            val_parquet=config.ORIGINAL_TRAIN_PARQUET.parent / "val.parquet",
            output_dir=config.CHECKPOINTS_DIR,
            n_capture_rows=len(captures_df),
            n_original_rows=len(pd.read_parquet(config.ORIGINAL_TRAIN_PARQUET)),
            max_epochs=max_epochs,
            on_epoch_complete=epoch_cb,
            cancel_event=record.cancel_event,
        )
    except Exception as e:  # noqa: BLE001
        _logger.exception("finetune job %s failed", record.job_id)
        jobs.transition(record, "failed", error=str(e))
        callbacks.broadcast_terminal(record, "job_failed")
        return

    with record.lock:
        record.new_checkpoint_path = result.get("checkpoint_path")
        cancelled_due_to_event = record.cancel_event.is_set()

    if cancelled_due_to_event:
        jobs.transition(record, "cancelled")
        callbacks.broadcast_terminal(record, "job_cancelled")
    else:
        jobs.transition(record, "complete")
        callbacks.broadcast_terminal(record, "job_complete")


@router.post("/finetune", response_model=JobAck)
async def start_finetune(req: FinetuneRequest) -> JobAck:
    base_checkpoint = _resolve_profile_path(req.base_profile_id)
    if base_checkpoint is None:
        raise HTTPException(status_code=404, detail="Base profile not found")

    captures_dir = _validate_captures_dir(req.captures_dir)

    # max_epochs is the finetune_model default (30); surface for snapshot only.
    max_epochs = 30

    record = jobs.create(
        kind="finetune",
        base_profile_id=req.base_profile_id,
        captures_dir=str(captures_dir),
        epochs_total=max_epochs,
    )
    record.loop = asyncio.get_running_loop()

    asyncio.create_task(_run_finetune_job(
        record=record,
        base_checkpoint=base_checkpoint,
        captures_dir=captures_dir,
        weight_recent=req.weight_recent,
        max_epochs=max_epochs,
    ))

    return JobAck(job_id=record.job_id, state=record.state)
