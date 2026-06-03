#!/usr/bin/env python
"""CLI wrapper for training a Sonna Editor Personal AI profile."""

from __future__ import annotations

from sonna_editor.training.profile_runner import _trainer_log_every_n_steps, main, train_profile

__all__ = ["_trainer_log_every_n_steps", "main", "train_profile"]


if __name__ == "__main__":
    main()
