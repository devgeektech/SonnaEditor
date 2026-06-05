from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import scripts.train_foundation_model as foundation_cli


def test_batch_size_attempts_halves_to_one() -> None:
    assert foundation_cli._batch_size_attempts(16) == [16, 8, 4, 2, 1]
    assert foundation_cli._batch_size_attempts(7) == [7, 3, 1]
    assert foundation_cli._batch_size_attempts(1) == [1]


def test_batch_size_attempts_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="batch-size"):
        foundation_cli._batch_size_attempts(0)


def test_is_cuda_memory_failure_matches_oom_and_cudnn_messages() -> None:
    assert foundation_cli._is_cuda_memory_failure(RuntimeError("CUDA error: out of memory"))
    assert foundation_cli._is_cuda_memory_failure(
        RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED_CUDART")
    )
    assert not foundation_cli._is_cuda_memory_failure(RuntimeError("plain data error"))


def test_train_profile_with_cuda_oom_retry_reduces_batch_size(monkeypatch) -> None:
    calls: list[int] = []

    def fake_train_profile(args: argparse.Namespace) -> dict:
        calls.append(args.batch_size)
        if args.batch_size == 16:
            raise RuntimeError("CUDA error: out of memory")
        return {"final_model": "ok.ckpt"}

    monkeypatch.setattr(foundation_cli, "train_profile", fake_train_profile)
    monkeypatch.setattr(foundation_cli, "_clear_cuda_after_failure", lambda: None)

    args = argparse.Namespace(batch_size=16)
    summary = foundation_cli._train_profile_with_cuda_oom_retry(args)

    assert calls == [16, 8]
    assert args.batch_size == 8
    assert summary["final_model"] == "ok.ckpt"
    assert summary["auto_reduced_batch_size_from"] == 16
    assert summary["auto_reduced_batch_size_to"] == 8


def test_train_profile_with_cuda_oom_retry_does_not_hide_non_cuda_errors(monkeypatch) -> None:
    def fake_train_profile(args: argparse.Namespace) -> dict:
        raise RuntimeError("bad parquet")

    monkeypatch.setattr(foundation_cli, "train_profile", fake_train_profile)

    with pytest.raises(RuntimeError, match="bad parquet"):
        foundation_cli._train_profile_with_cuda_oom_retry(argparse.Namespace(batch_size=16))


def test_foundation_parser_defaults_train_final_backbone_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_foundation_model.py",
            "--splits-dir",
            "data/training_workspace/fivek/splits_v2_stratified",
        ],
    )

    args = foundation_cli._parse_args()

    assert args.backbone_unfreeze_strategy == "progressive"
    assert args.backbone_trainable_layers == "stage:7"
    assert args.min_foundation_train_rows == 1000
    assert args.allow_small_foundation_dataset is False
    assert args.allow_quality_gate_failure is False


def test_validate_foundation_split_size_blocks_tiny_production_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        foundation_cli,
        "_split_row_counts",
        lambda splits_dir: {"train": 132, "val": 27, "test": 30},
    )

    with pytest.raises(RuntimeError, match="only 132 train rows"):
        foundation_cli._validate_foundation_split_size(
            splits_dir=tmp_path,
            min_train_rows=1000,
            allow_small_dataset=False,
        )


def test_validate_foundation_split_size_allows_explicit_small_ablation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        foundation_cli,
        "_split_row_counts",
        lambda splits_dir: {"train": 132, "val": 27, "test": 30},
    )

    counts = foundation_cli._validate_foundation_split_size(
        splits_dir=tmp_path,
        min_train_rows=1000,
        allow_small_dataset=True,
    )

    assert counts["train"] == 132


def test_foundation_quality_gate_flags_overfit_and_bad_metrics() -> None:
    summary = {
        "best_val_loss": 0.017,
        "test_results": {
            "test_loss": 0.028,
            "test_mae_exposure": 0.356,
            "test_mae_shadows": 38.9,
        },
    }

    failures = foundation_cli._foundation_quality_failures(summary)

    assert any("test_loss" in failure for failure in failures)
    assert any("Exposure2012" in failure for failure in failures)
    assert any("Shadows2012" in failure for failure in failures)
