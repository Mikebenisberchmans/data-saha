"""
conversation node: builds a system prompt from the user context loaded by
load_context (plus the rolling summary, if any), sends the full message
history to Groq, and appends the assistant's reply.

Phase 3 change: since AgentState.messages is now a plain list (see
app/agent/state.py), this node must return the COMPLETE new messages list
(existing + reply), not just the new message — there is no reducer doing
that merge for us anymore.

This node has NO knowledge of MCP tools or data sources yet — that's wired
in starting Phase 5 (source_selection) and Phase 6 (tool_execution).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.state import AgentState
from app.llm.groq_client import get_groq_client


def _build_system_prompt(state: AgentState) -> str:
    name = state.get("user_display_name") or "there"
    tz = state.get("user_timezone") or "UTC"
    summary = state.get("summary") or ""

    prompt = (
        "You are an AI analytics assistant for a desktop analytics agent. "
        f"You are speaking with {name} (timezone: {tz}). Address them by "
        "name naturally when it fits, but don't force it into every "
        "sentence. Be concise and direct. You do not yet have access to "
        "any data warehouses or MCP tools — that capability is being "
        "added in a later phase. If asked for data or analytics you can't "
        "access yet, say so plainly rather than inventing numbers."
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