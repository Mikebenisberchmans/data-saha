from __future__ import annotations

from datetime import datetime, timezone

from reportlab.graphics.shapes import Drawing

from app.mcp.models import NormalizedToolResult
from app.reports.generator import build_flowables, generate_report_pdf
from app.reports.specification import (
    QueryInfo,
    ReportChart,
    ReportMetric,
    ReportResult,
    ReportSpecification,
    ReportTable,
)


def _minimal_spec(**overrides) -> ReportSpecification:
    defaults = dict(
        title="Test Report",
        summary="A short summary.",
        key_findings=["Finding one.", "Finding two."],
        metrics=[ReportMetric(name="Total Revenue", value=1500.0, unit="USD")],
        tables=[
            ReportTable(
                title="Sales results",
                source_id="src-a",
                columns=["region", "revenue"],
                rows=[["US", 1000], ["EU", 500]],
            )
        ],
        charts=[],
        data_sources=["Sales Snowflake"],
        query_information=[QueryInfo(source_id="src-a", tool_name="execute_sql")],
        generation_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return ReportSpecification(**defaults)


def _dataset(source_id="src-a", columns=None, rows=None) -> NormalizedToolResult:
    return NormalizedToolResult(
        source_id=source_id,
        tool_name="execute_sql",
        columns=columns or ["region", "revenue"],
        rows=rows or [["US", 1000], ["EU", 500]],
    )


def test_build_flowables_includes_title():
    spec = _minimal_spec()
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    texts = [getattr(f, "text", "") for f in flowables]
    assert any("Test Report" in t for t in texts)


def test_build_flowables_omits_summary_section_when_empty():
    spec = _minimal_spec(summary="")
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    texts = [getattr(f, "text", "") for f in flowables]
    assert not any("Summary" == t for t in texts)


def test_build_flowables_includes_key_findings():
    spec = _minimal_spec()
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    texts = [getattr(f, "text", "") for f in flowables]
    assert any("Finding one." in t for t in texts)
    assert any("Finding two." in t for t in texts)


def test_build_flowables_includes_metrics_table():
    spec = _minimal_spec()
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    from reportlab.platypus import Table

    assert any(isinstance(f, Table) for f in flowables)


def test_build_flowables_includes_data_table_with_correct_rows():
    spec = _minimal_spec()
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    from reportlab.platypus import Table

    tables = [f for f in flowables if isinstance(f, Table)]
    assert len(tables) >= 2


def test_bar_chart_renders_as_drawing():
    chart = ReportChart(type="bar", title="Revenue by Region", x="region", y="revenue", source_id="src-a")
    spec = _minimal_spec(charts=[chart])
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    drawings = [f for f in flowables if isinstance(f, Drawing)]
    assert len(drawings) == 1


def test_line_chart_renders_as_drawing():
    chart = ReportChart(type="line", title="Revenue over Time", x="region", y="revenue", source_id="src-a")
    spec = _minimal_spec(charts=[chart])
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    drawings = [f for f in flowables if isinstance(f, Drawing)]
    assert len(drawings) == 1


def test_pie_chart_renders_as_drawing():
    chart = ReportChart(type="pie", title="Revenue Share", x="region", y="revenue", source_id="src-a")
    spec = _minimal_spec(charts=[chart])
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    drawings = [f for f in flowables if isinstance(f, Drawing)]
    assert len(drawings) == 1


def test_chart_with_missing_column_falls_back_gracefully():
    chart = ReportChart(type="bar", title="Bad Chart", x="nonexistent_col", y="revenue", source_id="src-a")
    spec = _minimal_spec(charts=[chart])
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    drawings = [f for f in flowables if isinstance(f, Drawing)]
    assert len(drawings) == 0
    texts = [getattr(f, "text", "") for f in flowables]
    assert any("could not be rendered" in t for t in texts)


def test_chart_referencing_unknown_source_falls_back_gracefully():
    chart = ReportChart(type="bar", title="Bad Chart", x="region", y="revenue", source_id="does-not-exist")
    spec = _minimal_spec(charts=[chart])
    result = ReportResult(specification=spec, datasets=[_dataset()])
    flowables = build_flowables(result)
    texts = [getattr(f, "text", "") for f in flowables]
    assert any("could not be rendered" in t for t in texts)


def test_generate_report_pdf_writes_a_real_pdf_file(tmp_path):
    chart = ReportChart(type="bar", title="Revenue by Region", x="region", y="revenue", source_id="src-a")
    spec = _minimal_spec(charts=[chart])
    result = ReportResult(specification=spec, datasets=[_dataset()])

    output_path = tmp_path / "report.pdf"
    returned_path = generate_report_pdf(result, output_path)

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    with open(output_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"


def test_generate_report_pdf_creates_parent_directories(tmp_path):
    spec = _minimal_spec()
    result = ReportResult(specification=spec, datasets=[_dataset()])

    output_path = tmp_path / "nested" / "dir" / "report.pdf"
    generate_report_pdf(result, output_path)

    assert output_path.exists()


def test_generate_report_pdf_handles_no_data_specification(tmp_path):
    """The 'no data available' fallback spec (from generate_report when
    nothing was gathered) must still render to a valid PDF, not crash."""
    spec = ReportSpecification(title="No data available", summary="Nothing found.")
    result = ReportResult(specification=spec, datasets=[])

    output_path = tmp_path / "empty_report.pdf"
    generate_report_pdf(result, output_path)

    assert output_path.exists()
    with open(output_path, "rb") as f:
        assert f.read(5) == b"%PDF-"