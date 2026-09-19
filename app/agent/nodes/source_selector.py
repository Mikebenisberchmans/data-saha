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

SOURCE_SELECTOR_SYSTEM_PROMPT = (
    "You are a routing component in an analytics assistant. Given the "
    "user's latest message, a short conversation summary (if any), and a "
    "list of configured data sources (each with an id, display name, "
    "provider, description, and business domain), decide which source(s), "
    "if any, are relevant to answering the message.\n\n"
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


def source_selector(state: AgentState) -> dict:
    profile = get_current_user_profile()
    repo = get_source_repository()

    candidates = _load_candidate_sources(profile.configured_data_sources, repo)
    if not candidates:
        return {"active_source_ids": []}

    user_message = _latest_user_message(state)
    if not user_message:
        return {"active_source_ids": []}

    candidates_payload = [s.selector_context() for s in candidates]
    user_content = (
        f"User message:\n{user_message}\n\n"
        f"Conversation summary so far (may be empty):\n{state.get('summary', '')}\n\n"
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

    return {"active_source_ids": filtered}