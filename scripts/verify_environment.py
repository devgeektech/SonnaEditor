#!/usr/bin/env python3
"""Verify the Sonna Editor development environment is correctly set up.

The project can run on macOS, Windows, and Linux. GPU checks are informative:
CUDA/MPS acceleration is preferred for training, but CPU-only machines are still
valid for development, tests, metadata work, and small inference runs.
"""

import platform
import sys

from sonna_editor.config import (
    APP_STATE_DIR,
    CHECKPOINTS_DIR,
    DNG_CONVERTER_ENV_VAR,
    DNG_CONVERTER_PATH,
    RAW_DIR,
    TRAINING_SOURCES_DIR,
    ensure_runtime_directories,
)
from sonna_editor.runtime import preferred_torch_device


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Print one pass/fail row and return the boolean for summary counting."""
    status = "OK" if ok else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
    """Run import, device, and optional external-tool checks for this host."""
    ensure_runtime_directories()
    print("\n=== Sonna Editor — Environment Verification ===\n")
    results = []

    # Python version
    ver = sys.version_info
    results.append(check(
        "Python version",
        ver >= (3, 11),
        f"Python {ver.major}.{ver.minor}.{ver.micro}",
    ))

    # Platform
    print(f"  [INFO] Platform: {platform.platform()}")
    print(f"  [INFO] Architecture: {platform.machine()}")
    print(f"  [INFO] Project root data dir: {RAW_DIR}")
    print(f"  [INFO] Training source root: {TRAINING_SOURCES_DIR}")
    print(f"  [INFO] Project root checkpoints dir: {CHECKPOINTS_DIR}")
    print(f"  [INFO] App state dir: {APP_STATE_DIR}")

    # PyTorch
    try:
        import torch
        results.append(check("PyTorch import", True, f"v{torch.__version__}"))

        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        device_name = preferred_torch_device()
        print(f"  [INFO] Preferred torch device: {device_name}")
        print(f"  [INFO] CUDA available: {cuda_available}")
        print(f"  [INFO] MPS available: {mps_available}")

        if device_name in {"cuda", "mps"}:
            device = torch.device(device_name)
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.matmul(a, b)
            results.append(check(
                f"{device_name.upper()} matmul (1000x1000)",
                c.shape == (1000, 1000),
                "tensor op succeeded on accelerator",
            ))
        else:
            a = torch.randn(256, 256)
            b = torch.randn(256, 256)
            c = torch.matmul(a, b)
            results.append(check(
                "CPU matmul (256x256)",
                c.shape == (256, 256),
                "CPU fallback is working",
            ))
    except ImportError as e:
        results.append(check("PyTorch import", False, str(e)))

    # Key dependencies
    for pkg in ["torchvision", "pytorch_lightning", "rawpy", "PIL", "lxml", "pandas", "pyarrow", "tqdm"]:
        try:
            __import__(pkg)
            results.append(check(f"{pkg} import", True))
        except ImportError:
            results.append(check(f"{pkg} import", False, "not installed"))

    # Adobe DNG Converter is optional for tests and UI development, but required
    # for RAW-to-DNG normalisation workflows. Report it without failing the
    # whole environment on machines that only do app/backend work.
    dng_path = DNG_CONVERTER_PATH
    dng_detail = str(dng_path) if dng_path.exists() else (
        f"not found; set {DNG_CONVERTER_ENV_VAR} or install Adobe DNG Converter"
    )
    print(f"  [INFO] Adobe DNG Converter: {dng_detail}")

    if dng_path.exists():
        import subprocess
        try:
            result = subprocess.run(
                [str(dng_path), "-version"],
                capture_output=True, text=True, timeout=10,
            )
            version_output = (result.stdout + result.stderr).strip().split("\n")[0]
            print(f"  [INFO] DNG Converter version output: {version_output}")
        except Exception:
            pass

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*48}")
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print("  Environment is ready for cross-platform development.")
    else:
        print("  Fix the FAIL items above before proceeding.")
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
