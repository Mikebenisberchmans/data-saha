"""
Central application configuration.

Every environment variable used anywhere in the project (across all phases)
is declared here so the settings shape is stable from Phase 1 onward, even
though most fields are unused until their corresponding phase is built.

Nothing secret-shaped lives in here as a *value* that gets passed around the
app — GROQ_API_KEY is the one exception (it's a platform credential, not a
per-source PAT, and is read directly by the Groq client in app/llm).
Per-source PATs are NEVER modeled here; they live only in a CredentialStore
(see app/core/credentials.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: Literal["development", "test", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    app_data_dir: Path = Field(default=Path("./data"), alias="APP_DATA_DIR")

    # --- Credential store ---
    credential_store_backend: Literal[
        "env", "local_file", "windows_credential_manager", "macos_keychain"
    ] = Field(default="env", alias="CREDENTIAL_STORE_BACKEND")
    local_credential_store_key: str | None = Field(
        default=None, alias="LOCAL_CREDENTIAL_STORE_KEY"
    )

    # --- Default local user (single-user desktop app) ---
    default_user_id: str = Field(default="local-user", alias="DEFAULT_USER_ID")
    default_user_display_name: str = Field(
        default="User", alias="DEFAULT_USER_DISPLAY_NAME"
    )
    default_user_timezone: str = Field(
        default="UTC", alias="DEFAULT_USER_TIMEZONE"
    )
    default_user_language: str = Field(default="en", alias="DEFAULT_USER_LANGUAGE")

    # --- Groq (Phase 2+) ---
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str | None = Field(default=None, alias="GROQ_MODEL")

    # --- Conversation memory (Phase 3+) ---
    summary_trigger_tokens: int = Field(
        default=6000, alias="SUMMARY_TRIGGER_TOKENS"
    )
    recent_messages_keep: int = Field(default=10, alias="RECENT_MESSAGES_KEEP")

    # --- SQL safety (Phase 7+) ---
    sql_safety_mode: Literal["strict", "permissive"] = Field(
        default="strict", alias="SQL_SAFETY_MODE"
    )

    @property
    def sources_file(self) -> Path:
        return self.app_data_dir / "sources.json"

    @property
    def user_profile_file(self) -> Path:
        return self.app_data_dir / "user_profile.json"

    @property
    def local_credential_store_file(self) -> Path:
        return self.app_data_dir / "credentials.local"
    @property
    def conversations_dir(self) -> Path:
        return self.app_data_dir / "conversations"

    def ensure_data_dir(self) -> None:
        self.app_data_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached settings accessor (simple singleton, no external deps)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force settings to be re-read (mainly useful in tests)."""
    global _settings
    _settings = Settings()
    return _settings
