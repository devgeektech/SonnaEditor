"""FastAPI app factory — mounts the /api routers and configures CORS."""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sonna_editor.api import jobs
from sonna_editor.api.routes import (
    captures,
    finetune,
    folders,
    health,
    process,
    profiles,
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Mark any persisted jobs that were "running" when the previous server
    # died as failed — a killed inference run is not safely resumable.
    jobs.recover_orphaned_jobs()
    yield

# Allow Electron renderer (file://) and any localhost origin.
_LOCALHOST_ORIGIN_RE = re.compile(r"^(https?://localhost(:\d+)?|file://.*)$")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sonna Editor API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_LOCALHOST_ORIGIN_RE.pattern,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(profiles.router, prefix="/api")
    app.include_router(folders.router, prefix="/api")
    app.include_router(captures.router, prefix="/api")
    app.include_router(process.router, prefix="/api")
    app.include_router(finetune.router, prefix="/api")

    return app


app = create_app()
