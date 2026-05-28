from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path

from tqdm import tqdm

from sonna_editor.config import SUPPORTED_RAW_EXTENSIONS
from sonna_editor.data.extract import extract_all
from sonna_editor.data.xmp import write_xmp
from sonna_editor.preset.adjuster import apply_adjustment, compute_adjustment
from sonna_editor.preset.parser import parse_preset

logger = logging.getLogger(__name__)


def _process_one(args: tuple) -> dict:
    """Worker: extract, adjust, write XMP for a single RAW file."""
    raw_path, preset, output_dir, options, dry_run = args
    result = {
        "raw_path": str(raw_path),
        "success": False,
        "output_xmp": None,
        "error": None,
    }
    try:
        data = extract_all(raw_path)
        image = data["preview"]
        metadata = {k: data.get(k) for k in
                    ("iso", "shutter_speed", "aperture", "focal_length",
                     "camera_body", "capture_datetime")}

        delta = compute_adjustment(image, metadata, preset, options)
        final_settings = apply_adjustment(preset, delta)

        if output_dir is not None:
            xmp_path = Path(output_dir) / (raw_path.stem + ".xmp")
        else:
            xmp_path = raw_path.with_suffix(".xmp")

        if not dry_run:
            write_xmp(xmp_path, final_settings, source_raw_path=raw_path)

        result["success"] = True
        result["output_xmp"] = str(xmp_path)
    except Exception as e:
        logger.error("Failed to process %s: %s", raw_path, e)
        result["error"] = str(e)

    return result


def process_shoot(
    input_dir: Path,
    output_dir: Path | None,
    preset_path: Path,
    options: dict,
    max_workers: int = 4,
    dry_run: bool = False,
) -> dict:
    """Apply a Lightroom preset (with content-aware adjustments) to a shoot.

    Walks input_dir for RAW files, computes per-photo adjustments, and writes
    XMP sidecars next to the originals (or into output_dir if given).

    Returns summary dict: {processed, failed, failures, output_paths}.
    """
    preset = parse_preset(preset_path)

    raw_files = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_RAW_EXTENSIONS
    )
    if not raw_files:
        raise ValueError(f"No RAW files found in {input_dir}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Processing %d RAW files from %s (dry_run=%s)",
        len(raw_files), input_dir, dry_run,
    )

    worker_args = [
        (raw, preset, output_dir, options, dry_run)
        for raw in raw_files
    ]

    results: list[dict] = []

    if max_workers == 1:
        iterator = (_process_one(a) for a in worker_args)
    else:
        pool = multiprocessing.Pool(processes=max_workers)
        iterator = pool.imap(_process_one, worker_args)

    try:
        for r in tqdm(iterator, total=len(raw_files), desc="Processing shoot", unit="photo"):
            results.append(r)
    finally:
        if max_workers != 1:
            pool.close()
            pool.join()

    processed = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    logger.info(
        "Done: %d processed, %d failed", len(processed), len(failed)
    )

    return {
        "processed": len(processed),
        "failed": len(failed),
        "failures": [{"path": r["raw_path"], "error": r["error"]} for r in failed],
        "output_paths": [r["output_xmp"] for r in processed],
    }
