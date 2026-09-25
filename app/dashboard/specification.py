"""
Dashboard specification generation (product spec sections 18, 19, 24).

The backend NEVER generates React/frontend code. It produces a structured
DashboardSpecification (title, description, charts, metrics, filters) plus
the normalized data each chart needs (NormalizedToolResult - spec section
22), and hands that as JSON to whatever frontend renders it (a future
React app using Plotly/ECharts/Recharts, per the product spec). This
module has no knowledge of any charting library.

`generate_dashboard()` is a plain callable, not something wired into the
main conversational graph's default path - per spec section 24 ("Python
should expose backend capabilities... chat(), generate_dashboard(),
generate_report()... do NOT write `if user_clicked_dashboard_button`").
The future FastAPI `POST /dashboard` route (Phase 11) will simply call
this directly.

Phase 10 change: data gathering itself moved to app/analytics/gathering.py
so app/reports/specification.py can reuse the exact same pipeline (spec
section 21's "avoid querying the warehouse twice unnecessarily") - this
module now only does dashboard-specific things: turning gathered data into
a DashboardSpecification.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from app.analytics.dataframe import combine_frames, frame_to_normalized_result, summarize_frame
from app.analytics.gathering import gather_datasets
from app.core.logging import get_logger
from app.llm.groq_client import get_groq_client
from app.mcp.models import NormalizedToolResult

logger = get_logger(__name__)


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
        specification=DashboardSpecification(title="No data available", description=reason),
        datasets=[],
    )


def generate_dashboard(question: str) -> DashboardResult:
    """Backend capability per spec section 24. Gathers data relevant to
    `question` from configured MCP sources, then asks the LLM to produce a
    DashboardSpecification grounded strictly in that gathered data (never
    invented columns/values), and returns it together with the normalized
    datasets a frontend needs to actually render it."""
    frames, _source_ids = gather_datasets(question, purpose="build a dashboard")
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