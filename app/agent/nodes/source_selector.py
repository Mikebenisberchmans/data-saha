"""
source_selector node: decides which configured data sources (if any) are
relevant to the user's latest message.

Per product spec section 8, this must not rely only on provider name —
multiple instances of the same provider (e.g. "Sales Snowflake" and
"Finance Snowflake") need to be distinguished using display_name,
description, and business_domain, all of which are the only fields passed
to the LLM (via DataSourceConfig.selector_context() — never mcp_url or
credential_ref). Tool discovery (already built in Phase 4) and actual MCP
tool calls (Phase 6) happen in later nodes; this node only decides WHICH
source(s), if any, are worth connecting to for the current message.

Phase 9 refactor: the core selection logic is exposed as
`select_sources_for_query`, a plain function taking a question string
rather than the full graph AgentState, so it can be reused by
app/dashboard/specification.py (which needs the same "which source(s)
match this request" decision outside of the conversational graph).
`source_selector` (the node) is now a thin adapter over it.

Bugfix (post-Phase 11, found in live testing): a bare follow-up like "in
that how many were from india" — after an earlier turn like "how many
users are there" — was being routed to `conversation` instead of
`tool_executor`, because this node only ever looked at the SINGLE latest
message plus the rolling summary (which is empty early in a conversation,
before the summarizer has triggered). With zero context, "in that how
many..." reads as a fragment with no obvious data need. Fixed by also
passing a short window of recent message history to the selector, and by
telling it explicitly to resolve references like "that"/"it"/"those
users" against that history rather than judging the latest message in
isolation.
"""

from __future__ import annotations

import json

from app.agent.state import AgentState
from app.core.logging import get_logger
from app.dependencies import get_current_user_profile, get_source_repository
from app.llm.groq_client import get_groq_client
from app.sources.models import DataSourceConfig
from app.sources.repository import SourceRepository

logger = get_logger(__name__)

# How many prior messages (not counting the latest one) to show the
# selector for resolving follow-up references. Kept small — this is a
# routing decision, not the full conversation context tool_executor gets.
RECENT_HISTORY_WINDOW = 6

SOURCE_SELECTOR_SYSTEM_PROMPT = (
    "You are a routing component in an analytics assistant. Given the "
    "user's latest message, a short conversation summary (if any), recent "
    "conversation history, and a list of configured data sources (each "
    "with an id, display name, provider, description, and business "
    "domain), decide which source(s), if any, are relevant to answering "
    "the LATEST message.\n\n"
    "Rules:\n"
    "- If the message is general conversation, a greeting, or doesn't need "
    "any specific data source, return an empty list.\n"
    "- If exactly one source is clearly relevant, return just that one.\n"
    "- If the question spans multiple sources (e.g. comparing data from "
    "two systems), return all of the relevant ones.\n"
    "- Use display_name, description, and business_domain to disambiguate "
    "between multiple sources of the same provider (e.g. 'Sales "
    "Snowflake' vs 'Finance Snowflake') — do not pick based on provider "
    "name alone.\n"
    "- The latest message is often a FOLLOW-UP that only makes sense "
    "given what came before — e.g. 'in that how many were from india' "
    "after 'how many users are there'. Use the recent conversation "
    "history to resolve words like 'that', 'it', 'those', or an implied "
    "subject. If an earlier turn in the shown history needed a source, "
    "and the latest message is clearly narrowing or continuing that same "
    "question, select that source again even though the latest message "
    "alone doesn't repeat the original keywords.\n"
    "- Only return ids that appear in the provided list. Never invent an "
    "id.\n\n"
    'Respond ONLY with a JSON object of the form {"source_ids": ["id1", '
    '"id2"]} and nothing else — no prose, no markdown fences.'
)


def _latest_user_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if msg.type != "ai":
            return msg.content
    return ""


def _recent_history_text(state: AgentState, window: int = RECENT_HISTORY_WINDOW) -> str:
    """Renders up to `window` messages BEFORE the latest user message, as
    plain "User: .../Assistant: ..." lines, for the selector to resolve
    follow-up references against. Excludes the latest message itself
    (that's passed separately as `question`)."""
    messages = state["messages"]
    if not messages:
        return ""
    history = messages[:-1] if messages else []
    recent = history[-window:]
    lines = []
    for msg in recent:
        role = "Assistant" if msg.type == "ai" else "User"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _load_candidate_sources(
    configured_ids: list[str], repo: SourceRepository
) -> list[DataSourceConfig]:
    sources = []
    for source_id in configured_ids:
        try:
            source = repo.get_source(source_id)
        except Exception:
            continue
        if source.enabled:
            sources.append(source)
    return sources


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def select_sources_for_query(
    question: str, summary: str = "", recent_history: str = ""
) -> list[str]:
    """Core selection logic, independent of the conversational graph.
    Used by the `source_selector` node below, and directly by
    app/dashboard/specification.py and app/reports/specification.py,
    which need the same source-matching decision without going through
    AgentState/graph.invoke() (and so pass recent_history="" — those are
    one-shot requests, not follow-ups within a running chat)."""
    profile = get_current_user_profile()
    repo = get_source_repository()

    candidates = _load_candidate_sources(profile.configured_data_sources, repo)
    if not candidates or not question:
        return []

    candidates_payload = [s.selector_context() for s in candidates]
    user_content = (
        f"Recent conversation history (oldest first, may be empty):\n"
        f"{recent_history or '(none)'}\n\n"
        f"Latest user message (the one to route):\n{question}\n\n"
        f"Conversation summary so far (may be empty):\n{summary}\n\n"
        f"Configured data sources:\n{json.dumps(candidates_payload, indent=2)}"
    )

    selected_ids: list[str] = []
    try:
        client = get_groq_client()
        response = client.chat(
            messages=[
                {"role": "system", "content": SOURCE_SELECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(_strip_json_fences(raw))
        selected_ids = parsed.get("source_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = []
    except Exception as exc:
        logger.warning("Source selection failed, defaulting to none: %s", exc)
        selected_ids = []

    valid_ids = {s.id for s in candidates}
    filtered = [sid for sid in selected_ids if sid in valid_ids]

    if filtered:
        logger.info("Source selector chose: %s", filtered)

    return filtered


def source_selector(state: AgentState) -> dict:
    question = _latest_user_message(state)
    history = _recent_history_text(state)
    filtered = select_sources_for_query(question, state.get("summary", ""), history)
    return {"active_source_ids": filtered}