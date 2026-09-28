"""
FastAPI entrypoint.

Phase 11: wires in all the API routes described in product spec section
23 - POST /chat, POST /dashboard, POST /report, GET /conversation/{id},
and GET/POST/DELETE /sources plus /sources/{id}/health. Every route is a
thin HTTP adapter over the backend capabilities already built in Phases
1-10 (app.agent.runner.run_turn, app.dashboard.specification.
generate_dashboard, app.reports.specification.generate_report, the
SourceRepository); no business logic lives in this file or in app/api/.

This module and everything under app/ must remain UI-independent - no
knowledge of Tauri, React, microphones, or desktop packaging belongs here.
The future Tauri app talks to this API exactly the same way scripts/*.py
and the test suite do.

Run with:
    python -m app.main
or:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import (
    routes_chat,
    routes_conversation,
    routes_dashboard,
    routes_report,
    routes_sources,
)
from app.config import get_settings
from app.core.logging import get_logger
from app.dependencies import get_current_user_profile, get_source_repository
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_data_dir()
    # Touch the repositories so config problems (e.g. a bad credential
    # backend) surface immediately on boot rather than on first request.
    get_source_repository()
    profile = get_current_user_profile()
    logger.info(
        "Backend started. env=%s user=%s configured_sources=%d",
        settings.app_env,
        profile.display_name,
        len(profile.configured_data_sources),
    )
    yield


app = FastAPI(
    title="AI Analytics Backend",
    description="UI-independent backend for a desktop AI analytics/voice agent.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["Content-Disposition"],  # lets the UI read the PDF filename
)

app.include_router(routes_chat.router)
app.include_router(routes_sources.router)
app.include_router(routes_dashboard.router)
app.include_router(routes_report.router)
app.include_router(routes_conversation.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/me")
def me() -> dict:
    """Returns the current user's profile (no secrets involved)."""
    profile = get_current_user_profile()
    return profile.model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)