from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import reload_settings
from app.dependencies import get_source_repository, get_user_profile_repository


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Points APP_DATA_DIR at a fresh tmp dir and clears cached singletons
    so each test gets a clean, isolated SourceRepository / settings.

    Also chdir's into tmp_path so pydantic-settings' relative `.env` lookup
    can never pick up a developer's real .env file (e.g. their real
    LOCAL_CREDENTIAL_STORE_KEY) — without this, tests that expect a
    variable to be *absent* can silently see the real project's .env
    contents instead, since pydantic-settings falls back to the .env file
    for any variable not present in the process environment."""
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CREDENTIAL_STORE_BACKEND", "env")
    monkeypatch.chdir(tmp_path)
    reload_settings()
    get_source_repository.cache_clear()
    get_user_profile_repository.cache_clear()
    yield tmp_path
    get_source_repository.cache_clear()
    get_user_profile_repository.cache_clear()