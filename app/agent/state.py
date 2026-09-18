"""
LangGraph state definition.

TypedDict (LangGraph's preferred shape) rather than a Pydantic model —
LangGraph nodes return *partial* state updates that get merged into the
running state, which fits TypedDict's structural nature better than
running full pydantic validation on every partial update.

CRITICAL invariant: nothing in this state may ever contain a PAT/secret.
Sources are referenced here by id only; any source metadata added in later
phases must go through DataSourceConfig.selector_context() (public fields
only), never mcp_url/credential_ref-resolved secrets.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    session_id: str

    # `add_messages` is LangGraph's reducer for message lists: instead of
    # each node's return value replacing `messages`, it's appended to it.
    messages: Annotated[list, add_messages]

    # Injected by load_context. Safe, non-secret user info only.
    user_display_name: str
    user_timezone: str
    preferred_language: str

    # Populated starting Phase 5 (source selection). Present now so the
    # state shape doesn't change again when that node is wired in.
    active_source_ids: list[str]

    last_response: str | None