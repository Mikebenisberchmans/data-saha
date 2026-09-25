"""
Shared data-gathering logic for non-chat backend capabilities
(generate_dashboard, generate_report — product spec section 24).

Extracted out of app/dashboard/specification.py in Phase 10 so
app/reports/specification.py can reuse the exact same source-selection +
MCP tool-calling + SQL-safety + normalization pipeline rather than
duplicating it, per spec section 21's "avoid querying the warehouse twice
unnecessarily" — one canonical code path for "go gather data relevant to
this request" that both capabilities call.

This deliberately does NOT generate any natural-language answer (that's
tool_executor.py's job on the chat path) — it only gathers and normalizes
data, stopping once the model indicates it has enough.
"""

from __future__ import annotations

import asyncio
import json

from app.agent.nodes.source_selector import select_sources_for_query
from app.analytics.dataframe import ToolResultFrame, summarize_frame, to_dataframe
from app.analytics.sql_validator import validate_sql
from app.core.logging import get_logger
from app.dependencies import get_mcp_manager
from app.llm.groq_client import get_groq_client
from app.mcp.models import ToolInfo

logger = get_logger(__name__)

MAX_DATA_GATHERING_ITERATIONS = 4

_SQL_ARGUMENT_KEYS = {"query", "sql", "statement", "sql_query", "sql_statement"}


def _tool_schema(tool: ToolInfo) -> dict:
    return {
        "type": "function",
        "function": {
            "name": f"{tool.source_id}__{tool.tool_name}",
            "description": tool.description or f"Tool on source {tool.source_id}",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    }


def _find_sql_argument_key(arguments: dict) -> str | None:
    for key, value in arguments.items():
        if key.lower() in _SQL_ARGUMENT_KEYS and isinstance(value, str):
            return key
    return None


async def _gather_datasets_async(
    question: str, purpose: str = "build a dashboard"
) -> tuple[list[ToolResultFrame], list[str]]:
    """Runs source selection + a bounded tool-calling loop to gather data
    relevant to `question`. `purpose` only changes the wording of the
    instruction given to the gathering model (e.g. "build a dashboard" vs
    "write a report") — the mechanics are identical either way.

    Returns (frames, source_ids) — source_ids is returned too so callers
    (e.g. reports, which need to name which sources were used) don't have
    to re-derive it from the frames alone (a source that was selected but
    returned no usable data would otherwise be invisible)."""
    source_ids = select_sources_for_query(question)
    if not source_ids:
        return [], []

    manager = get_mcp_manager()
    frames: list[ToolResultFrame] = []

    try:
        available_tools: list[ToolInfo] = []
        for sid in source_ids:
            try:
                available_tools.extend(await manager.discover_tools(sid))
            except Exception as exc:
                logger.warning("Data gathering: tool discovery failed for %s: %s", sid, exc)

        if not available_tools:
            return [], source_ids

        groq_tools = [_tool_schema(t) for t in available_tools]
        client = get_groq_client()
        gather_messages = [
            {
                "role": "system",
                "content": (
                    f"Call the available tools to gather the data needed to "
                    f"{purpose} for this request. Call as many tools as "
                    "needed across the available sources. Once you have "
                    "gathered enough data, stop calling tools."
                ),
            },
            {"role": "user", "content": question},
        ]

        for _ in range(MAX_DATA_GATHERING_ITERATIONS):
            response = client.chat(
                messages=gather_messages, tools=groq_tools, tool_choice="auto"
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                break

            gather_messages.append(
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
                source_id, _, tool_name = tc.function.name.partition("__")
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                sql_key = _find_sql_argument_key(arguments)
                if sql_key:
                    validation = validate_sql(arguments[sql_key])
                    if not validation.is_safe:
                        gather_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": (
                                    f"Query blocked by safety validation: "
                                    f"{validation.reason}"
                                ),
                            }
                        )
                        continue

                result = await manager.call_tool(source_id, tool_name, arguments)
                if not result.is_error:
                    frame = to_dataframe(result)
                    frames.append(frame)
                    content = summarize_frame(frame)
                else:
                    content = result.error or ""
                gather_messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": content or ""}
                )
    finally:
        for sid in reversed(source_ids):
            try:
                await manager.disconnect(sid)
            except asyncio.CancelledError:
                logger.warning("Data gathering: disconnect for %s cancelled", sid)
            except Exception:
                logger.warning("Data gathering: error disconnecting %s", sid)

    return frames, source_ids


def gather_datasets(
    question: str, purpose: str = "build a dashboard"
) -> tuple[list[ToolResultFrame], list[str]]:
    return asyncio.run(_gather_datasets_async(question, purpose))