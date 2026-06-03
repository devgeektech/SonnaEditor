from __future__ import annotations

import argparse

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
