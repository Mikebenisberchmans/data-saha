"""
conversation node: builds a system prompt from the user context loaded by
load_context (plus the rolling summary, and any sources the selector
identified as relevant), sends the full message history to Groq, and
appends the assistant's reply.

Phase 5 change: the system prompt now names which configured source(s), if
any, the source_selector node picked out for this message. This node still
does NOT call any MCP tool or fetch real data — that's Phase 6 (tool
execution) — so the prompt is explicit that identifying a relevant source
is not the same as having queried it yet.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.state import AgentState
from app.dependencies import get_source_repository
from app.llm.groq_client import get_groq_client


def _selected_source_names(state: AgentState) -> list[str]:
    source_ids = state.get("active_source_ids") or []
    if not source_ids:
        return []
    repo = get_source_repository()
    names = []
    for source_id in source_ids:
        try:
            names.append(repo.get_source(source_id).display_name)
        except Exception:
            continue
    return names


def _build_system_prompt(state: AgentState) -> str:
    name = state.get("user_display_name") or "there"
    tz = state.get("user_timezone") or "UTC"
    summary = state.get("summary") or ""
    selected_names = _selected_source_names(state)

    prompt = (
        "You are an AI analytics assistant for a desktop analytics agent. "
        f"You are speaking with {name} (timezone: {tz}). Address them by "
        "name naturally when it fits, but don't force it into every "
        "sentence. Be concise and direct."
    )

    if selected_names:
        joined = ", ".join(selected_names)
        prompt += (
            f"\n\nBased on this message, the source(s) '{joined}' look "
            "relevant. However, you still cannot query any data warehouse "
            "or MCP tool yet — that capability is being added in a later "
            "phase. Acknowledge which source(s) look relevant if it's "
            "useful context, but do not invent numbers or pretend to have "
            "run a query."
        )
    else:
        prompt += (
            "\n\nYou do not yet have access to any data warehouses or MCP "
            "tools — that capability is being added in a later phase. If "
            "asked for data or analytics you can't access yet, say so "
            "plainly rather than inventing numbers."
        )

    if summary:
        prompt += (
            "\n\nSummary of earlier parts of this conversation (the "
            "original messages have been dropped from active context to "
            "save space — treat this summary as ground truth for them):\n"
            f"{summary}"
        )
    return prompt


def _to_groq_messages(state: AgentState) -> list[dict]:
    groq_messages = [{"role": "system", "content": _build_system_prompt(state)}]
    for msg in state["messages"]:
        role = "assistant" if msg.type == "ai" else "user"
        groq_messages.append({"role": role, "content": msg.content})
    return groq_messages


def conversation(state: AgentState) -> dict:
    client = get_groq_client()
    groq_messages = _to_groq_messages(state)

    response = client.chat(messages=groq_messages)
    reply_text = response.choices[0].message.content or ""

    return {
        "messages": state["messages"] + [AIMessage(content=reply_text)],
        "last_response": reply_text,
    }