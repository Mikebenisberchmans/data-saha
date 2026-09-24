"""
Dashboard specification generation (product spec sections 18, 19, 24).

The backend NEVER generates React/frontend code. It produces a structured
DashboardSpecification (title, description, charts, metrics, filters) plus
the normalized data each chart needs (NormalizedToolResult — spec section
22), and hands that as JSON to whatever frontend renders it (a future
React app using Plotly/ECharts/Recharts, per the product spec). This
module has no knowledge of any charting library.

`generate_dashboard()` is a plain callable, not something wired into the
main conversational graph's default path — per spec section 24 ("Python
should expose backend capabilities... chat(), generate_dashboard(),
generate_report()... do NOT write `if user_clicked_dashboard_button`").
The future FastAPI `POST /dashboard` route (Phase 11) will simply call
this directly.

It reuses the same source-selection (Phase 5) and MCP tool-execution +
SQL safety + data normalization (Phases 4, 6, 7, 8) building blocks the
chat path uses, but ends in a structured JSON specification instead of
natural-language prose.
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.nodes.source_selector import select_sources_for_query
from app.analytics.dataframe import (
    ToolResultFrame,
    combine_frames,
    frame_to_normalized_result,
    summarize_frame,
    to_dataframe,
)
from app.analytics.sql_validator import validate_sql
from app.core.logging import get_logger
from app.dependencies import get_mcp_manager
from app.llm.groq_client import get_groq_client
from app.mcp.models import NormalizedToolResult, ToolInfo

logger = get_logger(__name__)

MAX_DATA_GATHERING_ITERATIONS = 4

_SQL_ARGUMENT_KEYS = {"query", "sql", "statement", "sql_query", "sql_statement"}


class ChartSpec(BaseModel):
    type: Literal["line", "bar", "pie", "area", "scatter", "table"]
    title: str
    x: str | None = None
    y: str | None = None
    source_id: str | None = Field(
        default=None,
        description="Which gathered dataset (by source_id) this chart's x/y "
        "columns refer to - lets the frontend bind the right dataset.",
    )


class MetricSpec(BaseModel):
    name: str
    value: float | str
    unit: str | None = None


class DashboardSpecification(BaseModel):
    title: str
    description: str = ""
    charts: list[ChartSpec] = Field(default_factory=list)
    metrics: list[MetricSpec] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)


class DashboardResult(BaseModel):
    """What generate_dashboard() returns: the specification PLUS the
    normalized data needed to actually render it (spec section 19). This
    is UI-independent - no chart-library-specific shape, just data + spec
    - and is exactly what a future FastAPI POST /dashboard response body
    would serialize."""

    specification: DashboardSpecification
    datasets: list[NormalizedToolResult] = Field(default_factory=list)


DASHBOARD_SYSTEM_PROMPT = (
    "You design analytics dashboard specifications from data that has "
    "already been gathered for you. Given the user's request and one or "
    "more normalized datasets (each with columns and sample rows), "
    "produce a dashboard specification as a JSON object with EXACTLY "
    "these fields: title (string), description (string), charts (array "
    "of objects with type, title, x, y, source_id), metrics (array of "
    "objects with name, value, unit), filters (array of column-name "
    "strings).\n\n"
    "type must be one of: line, bar, pie, area, scatter, table.\n"
    "x and y must be actual column names from the dataset referenced by "
    "source_id - never invent a column that isn't in the data given to "
    "you. Only propose charts and metrics the given data can actually "
    "support. If a metric requires simple aggregation (sum, average, "
    "count) of a column already shown, compute it yourself and report "
    "the resulting number - do not describe the aggregation in words.\n\n"
    "Respond with ONLY the JSON object - no prose, no markdown fences."
)


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


async def _gather_datasets_async(question: str) -> list[ToolResultFrame]:
    """Reuses Phase 5 source selection and Phase 4/6/7 MCP tool calling +
    SQL validation to gather data relevant to `question`, WITHOUT
    generating any natural-language answer - that's the difference from
    tool_executor.py's chat-path loop, which is why this isn't simply a
    call into that module."""
    source_ids = select_sources_for_query(question)
    if not source_ids:
        return []

    manager = get_mcp_manager()
    frames: list[ToolResultFrame] = []

    try:
        available_tools: list[ToolInfo] = []
        for sid in source_ids:
            try:
                available_tools.extend(await manager.discover_tools(sid))
            except Exception as exc:
                logger.warning("Dashboard: tool discovery failed for %s: %s", sid, exc)

        if not available_tools:
            return []

        groq_tools = [_tool_schema(t) for t in available_tools]
        client = get_groq_client()
        gather_messages = [
            {
                "role": "system",
                "content": (
                    "Call the available tools to gather the data needed to "
                    "build a dashboard for this request. Call as many tools "
                    "as needed across the available sources. Once you have "
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
        # Same LIFO-disconnect rule as tool_executor.py (Phase 8 bugfix):
        # anyio cancel scopes must be closed in reverse order of opening
        # within this single asyncio.run() task.
        for sid in reversed(source_ids):
            try:
                await manager.disconnect(sid)
            except asyncio.CancelledError:
                logger.warning("Dashboard: disconnect for %s cancelled", sid)
            except Exception:
                logger.warning("Dashboard: error disconnecting %s", sid)

    return frames


def _gather_datasets(question: str) -> list[ToolResultFrame]:
    return asyncio.run(_gather_datasets_async(question))


def _parse_specification(raw_json: str) -> DashboardSpecification:
    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return DashboardSpecification.model_validate(json.loads(cleaned.strip()))


def _no_data_result(reason: str) -> DashboardResult:
    return DashboardResult(
        specification=DashboardSpecification(
            title="No data available",
            description=reason,
        ),
        datasets=[],
    )


def generate_dashboard(question: str) -> DashboardResult:
    """Backend capability per spec section 24. Gathers data relevant to
    `question` from configured MCP sources, then asks the LLM to produce a
    DashboardSpecification grounded strictly in that gathered data (never
    invented columns/values), and returns it together with the normalized
    datasets a frontend needs to actually render it."""
    frames = _gather_datasets(question)
    tabular_frames = [f for f in frames if f.df is not None and not f.df.empty]

    if not tabular_frames:
        return _no_data_result(
            "I couldn't gather any usable tabular data for this request - "
            "either no configured source matched it, or the available "
            "tools didn't return data suitable for charting."
        )

    combined = combine_frames(tabular_frames)
    dataset_summaries = "\n\n".join(summarize_frame(f) for f in tabular_frames)
    user_content = f"Request: {question}\n\nGathered data:\n{dataset_summaries}"
    if combined:
        user_content += f"\n\n{combined}"

    client = get_groq_client()
    try:
        response = client.chat(
            messages=[
                {"role": "system", "content": DASHBOARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        spec = _parse_specification(raw)
    except Exception as exc:
        logger.warning("Dashboard specification generation/parsing failed: %s", exc)
        spec = DashboardSpecification(
            title="Dashboard generation failed",
            description=(
                "Data was gathered successfully, but a valid dashboard "
                "specification could not be produced from it."
            ),
        )

    datasets = [frame_to_normalized_result(f) for f in tabular_frames]
    return DashboardResult(specification=spec, datasets=datasets)