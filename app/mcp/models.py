"""
Normalized MCP data shapes.

These are the only MCP-related objects that should ever cross into the
agent layer (LangGraph nodes, later phases' source-selection/tool-execution
nodes) or get logged. They deliberately carry no connection details and no
secrets — just source_id (a public identifier) plus tool metadata/results.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ToolInfo(BaseModel):
    """Metadata for one tool discovered on one MCP source. Per product
    spec section 7, tools are discovered dynamically — never assumed —
    so this is populated only from what the server actually reports."""

    source_id: str
    tool_name: str
    description: str = ""
    input_schema: dict[str, Any] = {}


class ToolCallResult(BaseModel):
    """Normalized result of calling one MCP tool. `content` is the tool's
    text output (MCP tool results can contain multiple content blocks;
    non-text blocks are not yet handled — this is revisited if/when a
    configured source returns e.g. image content)."""

    source_id: str
    tool_name: str
    content: str
    is_error: bool = False
    error: str | None = None


class ConnectionHealth(BaseModel):
    source_id: str
    healthy: bool
    message: str | None = None
class NormalizedToolResult(BaseModel):
    """The normalized internal representation described in product spec
    section 22: source_id, tool_name, columns, rows, metadata,
    execution_time, error. Unlike app.analytics.dataframe.ToolResultFrame
    (which wraps a pandas DataFrame for computation), this is the
    plain-JSON, API/serialization-safe shape — what a dashboard or report
    payload actually carries to the future frontend. `columns`/`rows` are
    None when the underlying tool result wasn't tabular."""

    source_id: str
    tool_name: str
    columns: list[str] | None = None
    rows: list[list] | None = None
    metadata: dict[str, Any] = {}
    execution_time: float | None = None
    error: str | None = None