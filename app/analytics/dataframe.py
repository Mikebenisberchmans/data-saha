"""
Data normalization for MCP tool results (product spec sections 10, 22).

Different MCP sources return different shapes of data in a ToolCallResult's
`content` (a text string — see app/mcp/models.py). This module is where
that text gets turned into a normalized internal representation (a pandas
DataFrame when the content is tabular) so:

1. The LLM gets a clean, compact summary instead of an arbitrary raw blob
   (improves answer accuracy — the model reasons over structured facts,
   not a wall of JSON it has to parse itself).
2. When a turn pulls data from more than one source (per spec section 10 —
   "Compare sales revenue from Snowflake with customer churn from
   Redshift"), we can compute an actual cross-source comparison in Python
   BEFORE handing anything to the LLM, rather than hoping the model
   eyeballs two separate blobs correctly and does the arithmetic itself.

This module never assumes cross-database SQL joins are possible (per spec
section 10) — sources are queried independently (already true from Phase
6's tool loop) and combined here, in Python, after the fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from app.mcp.models import ToolCallResult
from app.mcp.models import NormalizedToolResult, ToolCallResult

@dataclass
class ToolResultFrame:
    """One tool result, normalized. `df` is None when the content wasn't
    tabular (e.g. a scalar value, free text, or malformed JSON) — callers
    fall back to the raw text in that case."""

    source_id: str
    tool_name: str
    df: pd.DataFrame | None
    raw_content: str


def to_dataframe(result: ToolCallResult) -> ToolResultFrame:
    """Attempts to interpret a tool result's content as tabular data.
    Recognized shapes (checked in order):
      - {"columns": [...], "rows": [[...], ...]}   (the common MCP shape)
      - a JSON list of objects, e.g. [{"region": "US", "revenue": 1000}]
      - a single JSON object, treated as one row
    Anything else (plain text, a scalar, malformed JSON) yields df=None
    rather than raising — normalization is best-effort, never fatal to the
    turn."""
    if result.is_error or not result.content:
        return ToolResultFrame(
            source_id=result.source_id,
            tool_name=result.tool_name,
            df=None,
            raw_content=result.content,
        )

    try:
        parsed = json.loads(result.content)
    except (json.JSONDecodeError, TypeError):
        return ToolResultFrame(
            source_id=result.source_id,
            tool_name=result.tool_name,
            df=None,
            raw_content=result.content,
        )

    df: pd.DataFrame | None = None
    try:
        if (
            isinstance(parsed, dict)
            and "columns" in parsed
            and "rows" in parsed
            and isinstance(parsed["columns"], list)
            and isinstance(parsed["rows"], list)
        ):
            df = pd.DataFrame(parsed["rows"], columns=parsed["columns"])
        elif isinstance(parsed, list) and parsed and all(
            isinstance(row, dict) for row in parsed
        ):
            df = pd.DataFrame(parsed)
        elif isinstance(parsed, dict):
            df = pd.DataFrame([parsed])
    except Exception:
        df = None

    return ToolResultFrame(
        source_id=result.source_id,
        tool_name=result.tool_name,
        df=df,
        raw_content=result.content,
    )


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="number").columns)


def summarize_frame(frame: ToolResultFrame, max_rows: int = 20) -> str:
    """Compact, LLM-friendly textual summary of one normalized tool
    result. Falls back to the raw content verbatim when normalization
    didn't produce a DataFrame."""
    if frame.df is None:
        return frame.raw_content

    df = frame.df
    lines = [f"Result from '{frame.tool_name}': {len(df)} row(s), columns: {list(df.columns)}"]

    preview = df.head(max_rows)
    lines.append(preview.to_string(index=False))
    if len(df) > max_rows:
        lines.append(f"... ({len(df) - max_rows} more row(s) not shown)")

    numeric_cols = _numeric_columns(df)
    if numeric_cols:
        stats = df[numeric_cols].agg(["sum", "mean", "min", "max"])
        lines.append("Numeric column stats:")
        lines.append(stats.to_string())

    return "\n".join(lines)


def combine_frames(frames: list[ToolResultFrame]) -> str | None:
    """Computes a cross-source comparison when results from more than one
    distinct source_id have successfully normalized into DataFrames.
    Returns None when there's nothing to combine (fewer than two sources
    with tabular data) — callers should skip injecting anything in that
    case.

    This is deliberately simple rather than a general analytics engine:
    for each source's frame, it totals the first numeric column found
    (a reasonable default for "revenue"/"count"-style analytics results)
    and reports source-by-source totals side by side, which is enough to
    answer "compare X from source A with Y from source B" style questions
    without assuming the two sources share a joinable key."""
    usable = [f for f in frames if f.df is not None and not f.df.empty]
    distinct_sources = {f.source_id for f in usable}
    if len(distinct_sources) < 2:
        return None

    lines = ["Cross-source summary (computed from the tool results above):"]
    for frame in usable:
        numeric_cols = _numeric_columns(frame.df)
        if numeric_cols:
            first_numeric = numeric_cols[0]
            total = frame.df[first_numeric].sum()
            lines.append(
                f"- {frame.source_id} ({frame.tool_name}): "
                f"{len(frame.df)} row(s), total {first_numeric} = {total}"
            )
        else:
            lines.append(
                f"- {frame.source_id} ({frame.tool_name}): "
                f"{len(frame.df)} row(s), no numeric column to total"
            )

    return "\n".join(lines)
def frame_to_normalized_result(
    frame: ToolResultFrame, execution_time: float | None = None
) -> NormalizedToolResult:
    """Converts an internal ToolResultFrame (pandas-based, used for
    computation) into the serializable NormalizedToolResult shape from
    product spec section 22 — what actually gets returned to a caller
    (e.g. a dashboard/report payload) rather than kept internal. When
    `frame.df` is None (not tabular), columns/rows are None and the raw
    text is preserved under metadata['raw_content'] instead."""
    if frame.df is None:
        return NormalizedToolResult(
            source_id=frame.source_id,
            tool_name=frame.tool_name,
            columns=None,
            rows=None,
            metadata={"raw_content": frame.raw_content},
            execution_time=execution_time,
        )

    return NormalizedToolResult(
        source_id=frame.source_id,
        tool_name=frame.tool_name,
        columns=list(frame.df.columns),
        rows=frame.df.values.tolist(),
        metadata={},
        execution_time=execution_time,
    )