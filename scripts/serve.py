#!/usr/bin/env python
"""Uvicorn entrypoint for the Sonna Editor API.

Usage:
    uv run scripts/serve.py [--port 8765] [--reload]

The server is hard-bound to 127.0.0.1. Anything else is rejected at startup —
the API is intended only for the local Electron shell, never for network exposure.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn
from sonna_editor import config

_ALLOWED_HOST = "127.0.0.1"


def main() -> None:
    config.ensure_runtime_directories()
    parser = argparse.ArgumentParser(description="Run the Sonna Editor API on localhost.")
    parser.add_argument("--host", default=_ALLOWED_HOST,
                        help=f"Bind host (must be {_ALLOWED_HOST}).")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default 8765).")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (dev only).")
    args = parser.parse_args()

    if args.host != _ALLOWED_HOST:
        print(
            f"Error: --host must be {_ALLOWED_HOST}; refusing to bind to {args.host!r}.",
            file=sys.stderr,
        )
        sys.exit(2)

    uvicorn.run(
        "sonna_editor.api.server:app",
        host=_ALLOWED_HOST,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
