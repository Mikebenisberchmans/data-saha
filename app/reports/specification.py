"""
Report specification generation (product spec section 20).

Report planning is kept separate from PDF rendering (that's Phase 10's
other half, app/reports/generator.py) - this module only produces a
ReportSpecification: title, summary, key findings, metrics, tables,
charts, data sources, query information, generation timestamp.

Design choice: not every field is LLM-generated. `data_sources`,
`query_information`, `generation_timestamp`, and `tables` are computed
deterministically from the data actually gathered - these are facts, and
trusting an LLM to restate facts it was already given risks it drifting
from them. Only `summary`, `key_findings`, and `charts` (interpretation
and presentation choices) go through Groq, with an explicit instruction to
stay grounded in the provided data - the same "never invent" discipline
used in app/dashboard/specification.py.

Reuses the shared gathering pipeline in app/analytics/gathering.py - the
same one generate_dashboard() uses - per spec section 21's "avoid
querying the warehouse twice unnecessarily."
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.analytics.dataframe import (
    ToolResultFrame,
    combine_frames,
    frame_to_normalized_result,
    summarize_frame,
)
from app.analytics.gathering import gather_datasets
from app.core.logging import get_logger
from app.dependencies import get_source_repository
from app.llm.groq_client import get_groq_client
from app.mcp.models import NormalizedToolResult

logger = get_logger(__name__)

MAX_TABLE_ROWS = 15


class ReportChart(BaseModel):
    type: Literal["line", "bar", "pie", "area", "scatter"]
    title: str
    x: str | None = None
    y: str | None = None
    source_id: str | None = None


class ReportMetric(BaseModel):
    name: str
    value: float | str
    unit: str | None = None


class ReportTable(BaseModel):
    title: str
    source_id: str
    columns: list[str]
    rows: list[list]


class QueryInfo(BaseModel):
    source_id: str
    tool_name: str


class ReportSpecification(BaseModel):
    title: str
    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    metrics: list[ReportMetric] = Field(default_factory=list)
    tables: list[ReportTable] = Field(default_factory=list)
    charts: list[ReportChart] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    query_information: list[QueryInfo] = Field(default_factory=list)
    generation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ReportResult(BaseModel):
    specification: ReportSpecification
    datasets: list[NormalizedToolResult] = Field(default_factory=list)


REPORT_SYSTEM_PROMPT = (
    "You write the narrative parts of an analytics report from data that "
    "has already been gathered for you. Given the user's request and one "
    "or more normalized datasets (each with columns and sample rows), "
    "produce a JSON object with EXACTLY these fields: title (string), "
    "summary (string, a short paragraph), key_findings (array of short "
    "strings, each one concrete finding), charts (array of objects with "
    "type, title, x, y, source_id).\n\n"
    "type must be one of: line, bar, pie, area, scatter.\n"
    "x and y must be actual column names from the dataset referenced by "
    "source_id - never invent a column, number, or finding that isn't "
    "supported by the data given to you. Every key finding must be "
    "traceable to something actually present in the data. Do not include "
    "tables, metrics, data_sources, query_information, or a timestamp in "
    "your JSON - those are added separately.\n\n"
    "Respond with ONLY the JSON object - no prose, no markdown fences."
)


def _resolve_source_names(source_ids: list[str]) -> list[str]:
    repo = get_source_repository()
    names = []
    for sid in source_ids:
        try:
            names.append(repo.get_source(sid).display_name)
        except Exception:
            names.append(sid)
    return names


def _frame_to_table(frame: ToolResultFrame) -> ReportTable:
    df = frame.df.head(MAX_TABLE_ROWS)
    return ReportTable(
        title=f"{frame.tool_name} results",
        source_id=frame.source_id,
        columns=list(df.columns),
        rows=df.values.tolist(),
    )


def _compute_metrics(frames: list[ToolResultFrame]) -> list[ReportMetric]:
    metrics: list[ReportMetric] = []
    for frame in frames:
        numeric_cols = list(frame.df.select_dtypes(include="number").columns)
        if numeric_cols:
            col = numeric_cols[0]
            total = frame.df[col].sum()
            metrics.append(
                ReportMetric(name=f"Total {col} ({frame.source_id})", value=float(total))
            )
    return metrics


def _parse_narrative(raw_json: str) -> dict:
    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def _no_data_spec(reason: str) -> ReportSpecification:
    return ReportSpecification(title="No data available", summary=reason)


def generate_report(question: str) -> ReportResult:
    frames, source_ids = gather_datasets(question, purpose="write a report")
    tabular_frames = [f for f in frames if f.df is not None and not f.df.empty]

    if not tabular_frames:
        return ReportResult(
            specification=_no_data_spec(
                "I couldn't gather any usable tabular data for this "
                "request - either no configured source matched it, or the "
                "available tools didn't return data suitable for a report."
            ),
            datasets=[],
        )

    data_source_names = _resolve_source_names(
        sorted({f.source_id for f in tabular_frames})
    )
    query_information = [
        QueryInfo(source_id=f.source_id, tool_name=f.tool_name) for f in tabular_frames
    ]
    tables = [_frame_to_table(f) for f in tabular_frames]
    metrics = _compute_metrics(tabular_frames)

    combined = combine_frames(tabular_frames)
    dataset_summaries = "\n\n".join(summarize_frame(f) for f in tabular_frames)
    user_content = f"Request: {question}\n\nGathered data:\n{dataset_summaries}"
    if combined:
        user_content += f"\n\n{combined}"

    client = get_groq_client()
    title, summary, key_findings, charts = "Report", "", [], []
    try:
        response = client.chat(
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = _parse_narrative(raw)
        title = parsed.get("title") or title
        summary = parsed.get("summary", "")
        key_findings = parsed.get("key_findings", []) or []
        charts = [ReportChart.model_validate(c) for c in parsed.get("charts", []) or []]
    except Exception as exc:
        logger.warning("Report narrative generation/parsing failed: %s", exc)
        summary = (
            "Data was gathered successfully, but the narrative summary "
            "could not be generated. See the tables and metrics below for "
            "the underlying data."
        )

    spec = ReportSpecification(
        title=title,
        summary=summary,
        key_findings=key_findings,
        metrics=metrics,
        tables=tables,
        charts=charts,
        data_sources=data_source_names,
        query_information=query_information,
    )

    datasets = [frame_to_normalized_result(f) for f in tabular_frames]
    return ReportResult(specification=spec, datasets=datasets)