from __future__ import annotations

from app.config import reload_settings


def test_settings_defaults(isolated_env):
    settings = reload_settings()
    assert settings.credential_store_backend == "env"
    assert settings.summary_trigger_tokens == 6000
    assert settings.sql_safety_mode == "strict"


def test_settings_reads_env_overrides(isolated_env, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("DEFAULT_USER_DISPLAY_NAME", "Mike")
    settings = reload_settings()
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.default_user_display_name == "Mike"


def test_sources_file_derived_from_data_dir(isolated_env):
    settings = reload_settings()
    assert settings.sources_file.parent == settings.app_data_dir
    assert settings.sources_file.name == "sources.json"
