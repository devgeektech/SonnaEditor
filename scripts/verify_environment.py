#!/usr/bin/env python3
"""Verify the Sonna Editor development environment is correctly set up."""

import platform
import sys
from pathlib import Path


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
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

    # PyTorch
    try:
        import torch
        results.append(check("PyTorch import", True, f"v{torch.__version__}"))

        mps_available = torch.backends.mps.is_available()
        results.append(check("MPS (Apple GPU) available", mps_available))

        if mps_available:
            device = torch.device("mps")
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.matmul(a, b)
            results.append(check(
                "MPS matmul (1000×1000)",
                c.shape == (1000, 1000),
                "tensor op succeeded on M1 GPU",
            ))
        else:
            results.append(check("MPS matmul", False, "skipped — MPS not available"))
    except ImportError as e:
        results.append(check("PyTorch import", False, str(e)))

    # Key dependencies
    for pkg in ["torchvision", "pytorch_lightning", "rawpy", "PIL", "lxml", "pandas", "pyarrow", "tqdm"]:
        try:
            __import__(pkg)
            results.append(check(f"{pkg} import", True))
        except ImportError:
            results.append(check(f"{pkg} import", False, "not installed"))

    # Adobe DNG Converter
    dng_path = Path("/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter")
    results.append(check(
        "Adobe DNG Converter",
        dng_path.exists(),
        str(dng_path) if dng_path.exists() else "not found at expected path",
    ))

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
        print("  Environment is ready. Proceed to Task 0.2 complete.")
    else:
        print("  Fix the FAIL items above before proceeding.")
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
