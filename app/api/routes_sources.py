"""
Source configuration endpoints (product spec section 23).

Mirrors scripts/manage_sources.py's behavior over HTTP - both go through
the same SourceRepository / UserProfileRepository, so a source created via
either path is visible to the other. Never returns a PAT: DataSourceConfig
has no secret field to begin with (see app/sources/models.py), so there is
nothing to redact here - the type system already guarantees it.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.dependencies import (
    get_current_user_profile,
    get_mcp_manager,
    get_source_repository,
    get_user_profile_repository,
)
from app.mcp.models import ConnectionHealth
from app.sources.models import DataSourceConfig, DataSourceCreate
from app.sources.repository import SourceNotFoundError

logger = get_logger(__name__)

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("", response_model=list[DataSourceConfig])
def list_sources() -> list[DataSourceConfig]:
    repo = get_source_repository()
    return repo.list_sources()


@router.post("", response_model=DataSourceConfig, status_code=201)
def create_source(data: DataSourceCreate) -> DataSourceConfig:
    repo = get_source_repository()
    created = repo.create_source(data)

    profile_repo = get_user_profile_repository()
    profile = get_current_user_profile()
    profile_repo.add_source(profile, created.id)

    return created


@router.delete("/{source_id}", status_code=204)
def delete_source(source_id: str) -> None:
    repo = get_source_repository()
    profile_repo = get_user_profile_repository()
    profile = get_current_user_profile()

    try:
        repo.delete_source(source_id)
    except SourceNotFoundError:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")

    profile_repo.remove_source(profile, source_id)


@router.get("/{source_id}/health", response_model=ConnectionHealth)
def source_health(source_id: str) -> ConnectionHealth:
    """Connects (if not already connected), checks health, then
    disconnects - a health probe shouldn't leave a lingering MCP
    connection open.

    Failing fast against an unreachable host has a known sharp edge here:
    the mcp SDK's Streamable HTTP transport can leave an async generator
    in a state that asyncio.run()'s own event-loop-shutdown phase (closing
    still-suspended async generators) raises a "cancel scope in a
    different task" RuntimeError for — AFTER our coroutine has already
    returned its ConnectionHealth result. That means no try/except *inside*
    the coroutine can catch it (see app/mcp/connection.py's health_check /
    _safe_close docstrings for the same failure mode at that layer); the
    only place left to guard is around asyncio.run() itself. The health
    result is captured via a mutable container so it survives even if
    asyncio.run() then raises during its own teardown."""
    manager = get_mcp_manager()
    captured: dict[str, ConnectionHealth] = {}

    async def _check() -> None:
        try:
            captured["health"] = await manager.health_check(source_id)
        finally:
            try:
                await manager.disconnect(source_id)
            except BaseException:
                pass

    try:
        asyncio.run(_check())
    except BaseException as exc:
        logger.warning(
            "asyncio.run() raised during health check teardown for %s: %s",
            source_id,
            type(exc).__name__,
        )

    return captured.get(
        "health",
        ConnectionHealth(
            source_id=source_id,
            healthy=False,
            message="Health check could not complete cleanly.",
        ),
    )