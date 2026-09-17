"""
Application-level dependency wiring.

Kept deliberately simple (module-level singletons) rather than a full DI
framework, since this is a single-process local desktop backend. Later
phases (MCPManager, ConversationStore, LangGraph app) get the same
treatment here so app/main.py and app/api/* stay thin.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.core.credentials import build_credential_store
from app.sources.models import UserProfile
from app.sources.repository import SourceRepository, UserProfileRepository


@lru_cache
def get_source_repository() -> SourceRepository:
    settings = get_settings()
    settings.ensure_data_dir()
    credential_store = build_credential_store()
    return SourceRepository(
        sources_file=settings.sources_file,
        credential_store=credential_store,
    )


@lru_cache
def get_user_profile_repository() -> UserProfileRepository:
    settings = get_settings()
    settings.ensure_data_dir()
    return UserProfileRepository(profile_file=settings.user_profile_file)


def get_current_user_profile() -> UserProfile:
    """Single-user desktop app: there's exactly one local profile. Multi-
    user support is out of scope per the product spec."""
    settings = get_settings()
    repo = get_user_profile_repository()
    default = UserProfile(
        user_id=settings.default_user_id,
        display_name=settings.default_user_display_name,
        timezone=settings.default_user_timezone,
        preferred_language=settings.default_user_language,
    )
    return repo.load(default=default)
