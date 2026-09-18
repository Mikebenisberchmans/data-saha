"""
load_context node: the first node in the graph. Injects the current user's
profile (display name, timezone, language) into state so downstream nodes
and prompts can use it without re-fetching it or manually re-adding it to
every prompt (per product spec section 4).
"""

from __future__ import annotations

from app.agent.state import AgentState
from app.dependencies import get_current_user_profile


def load_context(state: AgentState) -> dict:
    profile = get_current_user_profile()
    ctx = profile.system_context()
    return {
        "user_display_name": ctx["user_display_name"],
        "user_timezone": ctx["user_timezone"],
        "preferred_language": ctx["preferred_language"],
        # Carried through untouched until Phase 5 (source selection)
        # actually populates it.
        "active_source_ids": state.get("active_source_ids", []),
    }