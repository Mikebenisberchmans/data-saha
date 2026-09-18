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
from app.memory.conversation import ConversationStore
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

@lru_cache
def get_conversation_store() -> ConversationStore:
    settings = get_settings()
    settings.ensure_data_dir()
    return ConversationStore(conversations_dir=settings.conversations_dir)


def get_default_session_id() -> str:
    """The product has one continuous conversation per user (no multi-chat
    UI), so we use a stable, deterministic session id per user rather than
    a fresh random one per process — that's what actually makes
    persistence across restarts meaningful."""
    profile = get_current_user_profile()
    return f"{profile.user_id}-main"

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
