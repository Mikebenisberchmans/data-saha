from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import reload_settings
from app.dependencies import (
    get_conversation_store,
    get_source_repository,
    get_user_profile_repository,
)

@pytest.fixture(autouse=True)
def _fresh_settings_each_test():
    """Re-reads settings from the (by-now-reverted) real environment at the
    START of every test, isolated or not.

    Without this, get_settings()'s module-level cache can leak across
    tests: if test A calls reload_settings() (e.g. via the isolated_env
    fixture, or directly) with GROQ_API_KEY/GROQ_MODEL set to fake test
    values, that cached Settings object survives after test A's monkeypatch
    reverts the env vars — nothing re-reads it. A later test B that never
    asked for isolation (e.g. one asserting "GroqClient raises when
    GROQ_API_KEY is absent") would then see test A's fake key instead of
    a clean environment. Reloading here, before each test even starts (and
    therefore before that test's own monkeypatch.setenv calls run), closes
    that gap regardless of test order."""
    reload_settings()
    yield

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
    get_conversation_store.cache_clear()
    yield tmp_path
    get_source_repository.cache_clear()
    get_user_profile_repository.cache_clear()
    get_conversation_store.cache_clear()