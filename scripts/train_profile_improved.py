#!/usr/bin/env python
"""Train a Sonna Editor profile using the improved recipe defaults.

This wrapper applies safer, stronger defaults for the weak Temperature/Tint
fields and uses a higher-resolution input size for more accurate learning.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an improved Sonna Editor profile")
    p.add_argument("--train-parquet", required=True, type=Path, metavar="PATH")
    p.add_argument("--val-parquet",   required=True, type=Path, metavar="PATH")
    p.add_argument("--test-parquet",  required=True, type=Path, metavar="PATH")
    p.add_argument("--output-dir",    required=True, type=Path, metavar="DIR")
    p.add_argument("--max-epochs",    type=int, default=50)
    p.add_argument("--batch-size",    type=int, default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--freeze-backbone-epochs", type=int, default=3)
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--resume-from-checkpoint", type=Path, default=None, metavar="CKPT")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    python_exe = Path(sys.executable)
    script_path = Path(__file__).resolve().with_name("train_profile.py")

    cmd = [
        str(python_exe),
        str(script_path),
        "--train-parquet", str(args.train_parquet),
        "--val-parquet", str(args.val_parquet),
        "--test-parquet", str(args.test_parquet),
        "--output-dir", str(args.output_dir),
        "--max-epochs", str(args.max_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--freeze-backbone-epochs", str(args.freeze_backbone_epochs),
        "--num-workers", str(args.num_workers),
        "--image-resolution", "512",
        "--temperature-weight", "6.0",
        "--tint-weight", "6.0",
        "--exposure-weight", "2.0",
        "--temperature-bucket-loss-weight", "0.15",
        "--tint-bucket-loss-weight", "2.0",
        "--sign-wrong-penalty-weight", "0.2",
    ]
    if args.resume_from_checkpoint:
        cmd += ["--resume-from-checkpoint", str(args.resume_from_checkpoint)]

    print("Running improved training command:")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
