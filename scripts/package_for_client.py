#!/usr/bin/env python
"""Wrapper pointing to scripts/package_app.py for backwards compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_app import main

if __name__ == "__main__":
    main()
