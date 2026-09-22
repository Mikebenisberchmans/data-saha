"""
tool_executor node: runs the full MCP-backed agent/tool loop for a turn
where source_selector identified at least one relevant data source.

Per product spec section 15, this is NOT a single-shot
"LLM -> tool -> answer" pipeline: the model is given the ACTUAL tools
discovered on the selected source(s) (never assumed — see Phase 4's
dynamic discovery) and may call one, reason over the result, call another,
and so on, before producing its final answer. This node owns that whole
loop and produces the same output shape as the plain `conversation` node
(full messages list + last_response), so downstream (summarizer) doesn't
care which path a turn took.

Phase 7 addition: before any tool call is actually executed, its arguments
are checked for a SQL-shaped string (see _find_sql_argument_key). If found,
it goes through app/analytics/sql_validator.py — a real validation layer,
not just prompt instructions — before the MCP tool is ever called. A
rejected query never reaches the MCP server; the rejection reason is fed
back to the model as the tool's result, the same way a real execution
error would be, so the model can react (retry with a safe query, or tell
the user why it can't answer). This applies regardless of which MCP source
or tool is involved — the validator doesn't know or care about provider,
only about the SQL text itself.

LangGraph's compiled graph is invoked synchronously (graph.invoke, used by
app/agent/runner.py) but MCPManager's connect/discover/call methods are
async (Phase 4). Rather than convert the whole graph to async, this node
stays a plain sync function and drives its own asyncio event loop
internally via asyncio.run() — the simplest option given nothing else in
the current call chain is already inside an event loop.
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage

from app.agent.state import AgentState
from app.analytics.sql_validator import validate_sql
from app.core.logging import get_logger
from app.dependencies import get_mcp_manager, get_source_repository
from app.llm.groq_client import get_groq_client
from app.mcp.models import ToolInfo

logger = get_logger(__name__)

MAX_TOOL_ITERATIONS = 5

# Namespaces a tool name with its source id, since two different sources
# (e.g. two separate Snowflake instances) may both expose a tool called
# "execute_sql" — Groq's function-calling API needs globally unique tool
# names within one request.
_TOOL_NAME_SEP = "__"

# Argument key names commonly used by MCP tools that execute raw SQL.
# Deliberately provider-agnostic — per product spec section 16/29, we never
# assume a specific source exposes "execute_sql" specifically; we just
# check whatever argument keys the ACTUAL discovered tool schema used.
_SQL_ARGUMENT_KEYS = {"query", "sql", "statement", "sql_query", "sql_statement"}


def _namespaced_name(source_id: str, tool_name: str) -> str:
    return f"{source_id}{_TOOL_NAME_SEP}{tool_name}"


def _split_namespaced_name(namespaced: str) -> tuple[str, str]:
    source_id, _, tool_name = namespaced.partition(_TOOL_NAME_SEP)
    return source_id, tool_name


def _find_sql_argument_key(arguments: dict) -> str | None:
    for key, value in arguments.items():
        if key.lower() in _SQL_ARGUMENT_KEYS and isinstance(value, str):
            return key
    return None


def _tool_info_to_groq_schema(tool: ToolInfo) -> dict:
    return {
        "type": "function",
        "function": {
            "name": _namespaced_name(tool.source_id, tool.tool_name),
            "description": tool.description or f"Tool on source {tool.source_id}",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    }


def _selected_source_names(source_ids: list[str]) -> dict[str, str]:
    repo = get_source_repository()
    names = {}
    for sid in source_ids:
        try:
            names[sid] = repo.get_source(sid).display_name
        except Exception:
            names[sid] = sid
    return names


def _build_system_prompt(state: AgentState, source_names: dict[str, str]) -> str:
    name = state.get("user_display_name") or "there"
    tz = state.get("user_timezone") or "UTC"
    summary = state.get("summary") or ""
    sources_line = ", ".join(source_names.values())

    prompt = (
        "You are an AI analytics assistant for a desktop analytics agent. "
        f"You are speaking with {name} (timezone: {tz}). Address them by "
        "name naturally when it fits, but don't force it into every "
        "sentence. Be concise and direct.\n\n"
        f"You have live tool access to the following data source(s) for "
        f"this question: {sources_line}. Use the available tools to "
        "answer the user's question with real data. Call a tool whenever "
        "you need information you don't already have; you may call "
        "multiple tools, including more than once, before answering. Base "
        "your answer only on what the tools actually return — never "
        "invent numbers. If a tool call fails or returns no useful data, "
        "say so plainly rather than guessing. If you write SQL, only "
        "SELECT queries are permitted — any write or administrative "
        "statement will be blocked before it reaches the database."
    )
    if summary:
        prompt += f"\n\nSummary of earlier parts of this conversation:\n{summary}"
    return prompt


async def _run_tool_loop(state: AgentState) -> tuple[list, str]:
    source_ids = state["active_source_ids"]
    manager = get_mcp_manager()
    source_names = _selected_source_names(source_ids)

    try:
        return await _run_tool_loop_inner(state, source_ids, manager, source_names)
    finally:
        # IMPORTANT: each call to tool_executor() runs inside its own fresh
        # asyncio.run() loop (see tool_executor() below — this node is a
        # sync function driving async MCPManager/session calls). MCP
        # connections use anyio task groups/cancel scopes that are bound to
        # the loop they were opened in; if MCPManager kept a connection
        # alive for reuse across separate run_turn() calls, a later call's
        # *different* event loop would try to close/reuse it and hit
        # "attempted to exit cancel scope in a different task" errors. So
        # every turn tears down the connections it opened, within the same
        # loop, trading cross-turn connection reuse for correctness. Once
        # the whole app runs inside one long-lived event loop (Phase 11's
        # async FastAPI handlers), this can go back to being a persistent
        # pool.
        for sid in source_ids:
            try:
                await manager.disconnect(sid)
            except Exception:
                logger.warning("Error disconnecting source %s after turn", sid)


async def _run_tool_loop_inner(
    state: AgentState, source_ids: list[str], manager, source_names: dict[str, str]
) -> tuple[list, str]:
    # Discover tools across all selected sources (Phase 4 code). A source
    # that fails to connect/discover doesn't abort the whole turn — it's
    # just excluded from what the model is offered, per "handle connection
    # failures gracefully" (product spec section 6).
    available_tools: list[ToolInfo] = []
    for sid in source_ids:
        try:
            tools = await manager.discover_tools(sid)
            available_tools.extend(tools)
        except Exception as exc:
            logger.warning("Tool discovery failed for source %s: %s", sid, exc)

    if not available_tools:
        names = ", ".join(source_names.values()) or "the selected source(s)"
        final_text = (
            f"I couldn't reach any usable tools on {names} right now, so I "
            "can't pull live data for this. You could check the source's "
            "connection, or try again in a moment."
        )
        return state["messages"] + [AIMessage(content=final_text)], final_text

    groq_tools = [_tool_info_to_groq_schema(t) for t in available_tools]

    client = get_groq_client()
    groq_messages = [
        {"role": "system", "content": _build_system_prompt(state, source_names)}
    ]
    for msg in state["messages"]:
        role = "assistant" if msg.type == "ai" else "user"
        groq_messages.append({"role": role, "content": msg.content})

    final_text = ""
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat(
            messages=groq_messages, tools=groq_tools, tool_choice="auto"
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            final_text = message.content or ""
            break

        groq_messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            source_id, tool_name = _split_namespaced_name(tc.function.name)
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            sql_arg_key = _find_sql_argument_key(arguments)
            if sql_arg_key:
                validation = validate_sql(arguments[sql_arg_key])
                if not validation.is_safe:
                    logger.warning(
                        "Blocked unsafe SQL for tool %s on source %s: %s",
                        tool_name,
                        source_id,
                        validation.reason,
                    )
                    groq_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                f"Query blocked by safety validation: "
                                f"{validation.reason}"
                            ),
                        }
                    )
                    continue  # never reaches manager.call_tool

            result = await manager.call_tool(source_id, tool_name, arguments)
            content = result.error if result.is_error else result.content
            groq_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content or "",
                }
            )
    else:
        final_text = (
            "I gathered some data but wasn't able to finish reasoning about "
            "it within my step limit. Could you narrow down the question?"
        )
        logger.warning("Tool loop hit MAX_TOOL_ITERATIONS without finishing")

    return state["messages"] + [AIMessage(content=final_text)], final_text


def tool_executor(state: AgentState) -> dict:
    new_messages, final_text = asyncio.run(_run_tool_loop(state))
    return {"messages": new_messages, "last_response": final_text}