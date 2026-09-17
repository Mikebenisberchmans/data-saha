"""
FastAPI entrypoint.

Phase 1: only app setup, config validation on startup, and a couple of
introspection endpoints (health, current user, list sources) so the
skeleton is runnable and testable end-to-end before the agent/MCP layers
exist. Chat/dashboard/report routes are added starting Phase 11, per the
implementation plan.

Run with:
    python -m app.main
or:
    uvicorn app.main:app --reload

This module and everything under app/ must remain UI-independent — no
knowledge of Tauri, React, microphones, or desktop packaging belongs here.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings
from app.core.logging import get_logger
from app.dependencies import get_current_user_profile, get_source_repository

logger = get_logger(__name__)

app = FastAPI(
    title="AI Analytics Backend",
    description="UI-independent backend for a desktop AI analytics/voice agent.",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup() -> None:
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/me")
def me() -> dict:
    """Returns the current user's profile (no secrets involved)."""
    profile = get_current_user_profile()
    return profile.model_dump()


@app.get("/sources")
def list_sources() -> list[dict]:
    """Lists configured data sources. Never includes PATs — DataSourceConfig
    has no secret field, so there is nothing to redact here."""
    repo = get_source_repository()
    return [s.model_dump() for s in repo.list_sources()]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
