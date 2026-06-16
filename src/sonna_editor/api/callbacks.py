"""Callback bridges: sync ML loop → JobRecord → fan-out to WS subscribers.

These callbacks run inside `asyncio.to_thread` workers (per-photo) or inside
PyTorch Lightning's main training thread (per-epoch). They MUST:
  - never block on I/O that could deadlock the loop they're running in
  - never raise — every call is wrapped in try/except by the caller
  - be cheap enough that a slow callback doesn't dominate inference time

Broadcast safety: the websocket fan-out uses ``loop.call_soon_threadsafe``
to schedule ``queue.put_nowait`` on each per-subscriber asyncio.Queue. The
queues are bounded; on QueueFull the subscriber is dropped (slow-consumer
policy) so a paused client cannot grow memory unboundedly.

edit_summary uses fixed slots (Exp + WB + rotating tone slider) to match
the UI mock exactly — see SAHA UI/direction-console-v2.jsx LOG array.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Callable, Optional

import torch

from sonna_editor.api import jobs
from sonna_editor.api.confidence import scalar_confidence

_logger = logging.getLogger(__name__)


# ── Broadcast helpers ──────────────────────────────────────────────────────

def _broadcast(record: jobs.JobRecord, message: dict[str, Any]) -> None:
    """Push `message` to every subscriber Queue, drop slow consumers.

    Safe to call from any thread. Uses loop.call_soon_threadsafe so the
    actual put_nowait runs on the API loop's thread.
    """
    loop = record.loop
    if loop is None:
        return  # no event loop captured; subscribers cannot exist

    with record.lock:
        subscribers = list(record.subscribers)

    for queue in subscribers:
        def _put(q: asyncio.Queue = queue, msg: dict[str, Any] = message) -> None:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                with record.lock:
                    if q in record.subscribers:
                        record.subscribers.remove(q)
                _logger.info("dropped slow subscriber for job %s", record.job_id)

        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            # API loop is shutting down — broadcast is best-effort.
            pass


def broadcast_terminal(record: jobs.JobRecord, message_type: str) -> None:
    """Emit a terminal websocket message (job_complete/cancelled/failed)."""
    with record.lock:
        snap = record.snapshot_dict()
    payload = {"type": message_type, **snap}
    _broadcast(record, payload)


# ── Edit summary formatter ─────────────────────────────────────────────────

# Per the approved plan: fixed Exp + WB + one rotating tone slider.
_TONE_CANDIDATES: list[tuple[str, str]] = [
    ("Shadows2012", "Shad"),
    ("Highlights2012", "High"),
    ("Contrast2012", "Cont"),
    ("Vibrance", "Vib"),
    ("Saturation", "Sat"),
    ("Tint", "Tint"),
]


def _slider_float(value: Any, default: float) -> float:
    """Return a finite slider value, falling back for sparse Lite payloads."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _format_edit_summary(predicted: dict[str, float | None]) -> str:
    """Build the Live-log line: 'Exp +0.42 · WB 5,180K · Shad +38'."""
    exp = _slider_float(predicted.get("Exposure2012"), 0.0)
    temp = _slider_float(predicted.get("Temperature"), 5500.0)

    # Pick the largest-abs-value tone slider for the third slot
    third_field, third_abbr = max(
        _TONE_CANDIDATES,
        key=lambda fa: abs(_slider_float(predicted.get(fa[0]), 0.0)),
    )
    third_val = int(round(_slider_float(predicted.get(third_field), 0.0)))

    return (
        f"Exp {exp:+.2f} · "
        f"WB {int(round(temp)):,}K · "
        f"{third_abbr} {third_val:+d}"
    )


# ── Per-photo callback factory ─────────────────────────────────────────────

def make_photo_prepared_callback(
    record: jobs.JobRecord,
    started_at: float,
) -> Callable[[dict[str, Any]], None]:
    """Build the preview/metadata-prepared callback for live early progress."""
    def cb(photo: dict[str, Any]) -> None:
        try:
            with record.lock:
                record.photos_prepared += 1
                record.current_photo = photo["name"]
                elapsed = max(time.monotonic() - started_at, 1e-6)
                progress_units = max(record.photos_prepared, record.photos_processed)
                record.photos_per_sec = round(progress_units / elapsed, 1)
                remaining = (record.photos_total or progress_units) - progress_units
                record.eta_seconds = (
                    int(remaining / record.photos_per_sec)
                    if record.photos_per_sec > 0 and remaining > 0
                    else 0
                )
                prepared_now = record.photos_prepared
                total_now = record.photos_total
                pps_now = record.photos_per_sec
                eta_now = record.eta_seconds

            jobs.note_progress(record)

            _broadcast(record, {
                "type": "photo_prepared",
                "name": photo["name"],
                "photos_total": total_now,
                "photos_prepared": prepared_now,
                "photos_per_sec": pps_now,
                "eta_seconds": eta_now,
            })
        except Exception as e:  # noqa: BLE001
            _logger.warning("photo prepared callback failed for job %s: %s",
                            record.job_id, e)

    return cb


def make_photo_callback(
    record: jobs.JobRecord,
    started_at: float,
) -> Callable[[dict[str, Any]], None]:
    """Build the on_photo_complete callback for one process job.

    The callback runs in the asyncio.to_thread worker — it must remain sync
    and cheap. It updates the JobRecord snapshot fields, persists every Nth
    update via jobs.note_progress(), and broadcasts a photo_complete message.
    """
    def cb(photo: dict[str, Any]) -> None:
        try:
            std = photo.get("std")
            confidence: Optional[float] = None
            if isinstance(std, torch.Tensor):
                confidence = scalar_confidence(std)

            with record.lock:
                record.photos_processed += 1
                record.current_photo = photo["name"]
                if photo["status"] == "flag":
                    record.photos_flagged += 1
                xmp_path = photo.get("xmp_path")
                if xmp_path:
                    record.output_paths_so_far.append(str(xmp_path))

                elapsed = max(time.monotonic() - started_at, 1e-6)
                record.photos_per_sec = round(record.photos_processed / elapsed, 1)
                remaining = (record.photos_total or record.photos_processed) - record.photos_processed
                record.eta_seconds = (
                    int(remaining / record.photos_per_sec)
                    if record.photos_per_sec > 0 and remaining > 0
                    else 0
                )
                processed_now = record.photos_processed
                total_now = record.photos_total
                pps_now = record.photos_per_sec
                eta_now = record.eta_seconds

            jobs.note_progress(record)

            ws_msg: dict[str, Any] = {
                "type": "photo_complete",
                "name": photo["name"],
                "edit_summary": _format_edit_summary(photo["predicted_values"]),
                "status": photo["status"],
                "photos_total": total_now,
                "photos_processed": processed_now,
                "photos_per_sec": pps_now,
                "eta_seconds": eta_now,
            }
            if confidence is not None:
                ws_msg["confidence"] = round(confidence, 3)
            _broadcast(record, ws_msg)
        except Exception as e:  # noqa: BLE001 — never crash inference
            _logger.warning("photo callback failed for job %s: %s",
                            record.job_id, e)

    return cb


# ── Per-epoch callback factory ─────────────────────────────────────────────

def make_epoch_callback(
    record: jobs.JobRecord,
) -> Callable[[dict[str, Any]], None]:
    """Build the on_epoch_complete callback for one finetune job.

    Called from Lightning's main training thread inside
    on_validation_epoch_end. Receives {"epoch", "train_loss", "val_loss"}.
    """
    def cb(epoch_info: dict[str, Any]) -> None:
        try:
            with record.lock:
                record.epochs_completed += 1
                record.current_epoch = int(epoch_info["epoch"])
                tl = epoch_info.get("train_loss")
                vl = epoch_info.get("val_loss")
                record.train_loss = float(tl) if tl is not None else None
                record.val_loss = float(vl) if vl is not None else None
                snap_epoch = record.current_epoch
                snap_tl = record.train_loss
                snap_vl = record.val_loss

            jobs.note_progress(record)

            _broadcast(record, {
                "type": "epoch_complete",
                "epoch": snap_epoch,
                "train_loss": snap_tl,
                "val_loss": snap_vl,
            })
        except Exception as e:  # noqa: BLE001
            _logger.warning("epoch callback failed for job %s: %s",
                            record.job_id, e)

    return cb
