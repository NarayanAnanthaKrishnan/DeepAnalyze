from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import settings
from .routers.chat import router as chat_router
from .routers.export import router as export_router
from .routers.gate import router as gate_router
from .routers.workspace import router as workspace_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.gate_enforced and not settings.gate_active:
        log.warning(
            "GATE_ENFORCED is true but Supabase credentials are missing — "
            "the trial gate is INACTIVE (all requests pass)."
        )
    elif settings.gate_active:
        log.info("Trial gate active (Supabase: %s)", settings.supabase_url)
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,  # auth via Authorization header, not cookies
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "X-Admin-Key"],
    )
    app.include_router(gate_router)
    app.include_router(workspace_router)
    app.include_router(chat_router)
    app.include_router(export_router)
    return app


app = create_app()
