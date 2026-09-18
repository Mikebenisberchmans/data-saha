"""
LangGraph state definition.

Phase 3 change from Phase 2: `messages` is now a plain list (default
"last write wins" reducer), not `Annotated[list, add_messages]`. Reason:
the summarizer node needs to *replace* the message list (drop older
messages, keep only the most recent N) as well as append to it, and full
control of the list is simpler than expressing "replace" via
`add_messages` + `RemoveMessage` markers. This also aligns with Phase 3's
persistence model: conversation history now lives on disk (see
app/memory/conversation.py + app/agent/runner.py) rather than in a
LangGraph checkpointer, so every graph.invoke() is given the complete
message history as input and every node that touches `messages` must
return the complete new list, not a delta.

CRITICAL invariant (unchanged from Phase 1/2): nothing in this state may
ever contain a PAT/secret. Sources are referenced here by id only.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    session_id: str
    messages: list  # each node that changes this returns the FULL new list

    # Injected by load_context. Safe, non-secret user info only.
    user_display_name: str
    user_timezone: str
    preferred_language: str

    # Populated starting Phase 5 (source selection).
    active_source_ids: list[str]

    # Rolling summary of dropped older messages (Phase 3, product spec
    # section 13). Empty string until the first summarization trigger.
    summary: str

    last_response: str | None