from __future__ import annotations

import logging
import subprocess
from multiprocessing.pool import Pool
from pathlib import Path

from tqdm import tqdm

from sonna_editor.config import DNG_CONVERTER_PATH, SUPPORTED_RAW_EXTENSIONS

logger = logging.getLogger(__name__)


class DNGConverterNotFoundError(FileNotFoundError):
    pass


class DNGConversionError(RuntimeError):
    pass


def _check_binary() -> None:
    if not DNG_CONVERTER_PATH.exists():
        raise DNGConverterNotFoundError(
            f"Adobe DNG Converter not found at {DNG_CONVERTER_PATH}. "
            "Install from https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html"
        )


def get_dng_converter_version() -> str:
    """Return the installed Adobe DNG Converter version string."""
    _check_binary()
    result = subprocess.run(
        [str(DNG_CONVERTER_PATH), "-version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    # DNG Converter prints version to stdout or stderr depending on version
    output = (result.stdout + result.stderr).strip()
    return output if output else "unknown"


def convert_to_dng(
    input_path: Path,
    output_dir: Path,
    embed_original: bool = False,
    compress: bool = True,
) -> Path:
    """Convert a single RAW file to DNG using Adobe DNG Converter.

    Returns the path to the output DNG file.
    Raises DNGConverterNotFoundError if the binary is missing.
    Raises DNGConversionError if conversion fails.
    Raises ValueError if the input format is not supported.
    """
    _check_binary()

    if input_path.suffix.lower() not in SUPPORTED_RAW_EXTENSIONS:
        raise ValueError(
            f"Unsupported RAW format: {input_path.suffix}. "
            f"Supported: {SUPPORTED_RAW_EXTENSIONS}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(DNG_CONVERTER_PATH)]
    if compress:
        cmd.append("-c")
    if embed_original:
        cmd.append("-e")
    cmd += ["-d", str(output_dir), str(input_path)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise DNGConversionError(
            f"DNG Converter failed for {input_path.name} "
            f"(exit {result.returncode}): {result.stderr.strip()}"
        )

    output_path = output_dir / (input_path.stem + ".dng")
    if not output_path.exists():
        raise DNGConversionError(
            f"Conversion appeared to succeed but output not found: {output_path}"
        )

    return output_path


def _convert_worker(args: tuple[Path, Path, bool, bool]) -> Path | None:
    """Worker function for multiprocessing — returns None on failure."""
    input_path, output_dir, embed_original, compress = args
    try:
        return convert_to_dng(input_path, output_dir, embed_original, compress)
    except Exception as exc:
        logger.error("Failed to convert %s: %s", input_path.name, exc)
        return None


def batch_convert(
    input_paths: list[Path],
    output_dir: Path,
    max_workers: int = 4,
    embed_original: bool = False,
    compress: bool = True,
) -> list[Path | None]:
    """Convert multiple RAW files to DNG in parallel.

    Returns a list parallel to input_paths — each entry is the output Path on
    success or None on failure. Failures are logged to output_dir/failures.log.
    """
    _check_binary()
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        (p, output_dir, embed_original, compress)
        for p in input_paths
    ]

    results: list[Path | None] = []
    failures: list[str] = []

    with Pool(processes=max_workers) as pool:
        for input_path, result in zip(
            input_paths,
            tqdm(
                pool.imap(_convert_worker, args),
                total=len(args),
                desc="Converting to DNG",
                unit="file",
            ),
        ):
            results.append(result)
            if result is None:
                failures.append(str(input_path))

    if failures:
        failure_log = output_dir / "failures.log"
        failure_log.write_text("\n".join(failures) + "\n")
        logger.warning(
            "%d/%d conversions failed. See %s",
            len(failures),
            len(input_paths),
            failure_log,
        )

    return results
