"""
PDF report generation (product spec section 21).

ReportSpecification (app/reports/specification.py) -> this module -> PDF.
Deliberately modular: one function builds each section of the document
(title/header, summary, key findings, metrics, tables, charts), and
`generate_report_pdf` composes them. Each builder function returns a list
of ReportLab "flowables" (or a single Drawing for charts) so sections can
be tested independently without writing a file to disk.

Charts are rendered natively via reportlab.graphics (bar/line/pie), built
directly from the same normalized datasets already gathered for the
report — no matplotlib or other charting dependency, and no re-querying
the warehouse (spec section 21's "avoid querying the warehouse twice").
A chart type/data combination ReportLab's graphics package doesn't support
here (scatter, or a chart whose x/y columns can't be found in the
matching dataset) falls back to a small note in the PDF rather than
silently dropping it or failing the whole document.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.logging import get_logger
from app.mcp.models import NormalizedToolResult
from app.reports.specification import ReportChart, ReportResult, ReportSpecification

logger = get_logger(__name__)

_STYLES = getSampleStyleSheet()
_CHART_WIDTH = 420
_CHART_HEIGHT = 200
MAX_TABLE_ROWS_IN_PDF = 20


def _build_header(spec: ReportSpecification) -> list:
    flowables = [Paragraph(spec.title, _STYLES["Title"])]
    timestamp_style = ParagraphStyle(
        "Timestamp", parent=_STYLES["Normal"], textColor=colors.grey, fontSize=9
    )
    flowables.append(
        Paragraph(
            f"Generated {spec.generation_timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            timestamp_style,
        )
    )
    if spec.data_sources:
        flowables.append(
            Paragraph(f"Data sources: {', '.join(spec.data_sources)}", timestamp_style)
        )
    flowables.append(Spacer(1, 0.25 * inch))
    return flowables


def _build_summary(spec: ReportSpecification) -> list:
    if not spec.summary:
        return []
    flowables = [Paragraph("Summary", _STYLES["Heading2"])]
    flowables.append(Paragraph(spec.summary, _STYLES["BodyText"]))
    flowables.append(Spacer(1, 0.2 * inch))
    return flowables


def _build_key_findings(spec: ReportSpecification) -> list:
    if not spec.key_findings:
        return []
    flowables = [Paragraph("Key Findings", _STYLES["Heading2"])]
    for finding in spec.key_findings:
        flowables.append(Paragraph(f"\u2022 {finding}", _STYLES["BodyText"]))
    flowables.append(Spacer(1, 0.2 * inch))
    return flowables


def _build_metrics_table(spec: ReportSpecification) -> list:
    if not spec.metrics:
        return []
    flowables = [Paragraph("Metrics", _STYLES["Heading2"])]
    data = [["Metric", "Value", "Unit"]]
    for m in spec.metrics:
        value_str = f"{m.value:,.2f}" if isinstance(m.value, float) else str(m.value)
        data.append([m.name, value_str, m.unit or ""])
    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
            ]
        )
    )
    flowables.append(table)
    flowables.append(Spacer(1, 0.2 * inch))
    return flowables


def _build_data_tables(spec: ReportSpecification) -> list:
    if not spec.tables:
        return []
    flowables = [Paragraph("Data Tables", _STYLES["Heading2"])]
    for rt in spec.tables:
        flowables.append(Paragraph(rt.title, _STYLES["Heading3"]))
        rows = rt.rows[:MAX_TABLE_ROWS_IN_PDF]
        data = [rt.columns] + [[str(v) for v in row] for row in rows]
        table = Table(data, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        flowables.append(table)
        if len(rt.rows) > MAX_TABLE_ROWS_IN_PDF:
            flowables.append(
                Paragraph(
                    f"... ({len(rt.rows) - MAX_TABLE_ROWS_IN_PDF} more row(s) not shown)",
                    _STYLES["Italic"],
                )
            )
        flowables.append(Spacer(1, 0.15 * inch))
    return flowables


def _find_dataset(
    chart: ReportChart, datasets: list[NormalizedToolResult]
) -> NormalizedToolResult | None:
    candidates = [
        d for d in datasets
        if d.source_id == chart.source_id and d.columns and d.rows
    ] if chart.source_id else [d for d in datasets if d.columns and d.rows]
    return candidates[0] if candidates else None


def _extract_xy(
    dataset: NormalizedToolResult, chart: ReportChart
) -> tuple[list[str], list[float]] | None:
    if not chart.x or not chart.y or chart.x not in dataset.columns or chart.y not in dataset.columns:
        return None
    x_idx = dataset.columns.index(chart.x)
    y_idx = dataset.columns.index(chart.y)
    labels: list[str] = []
    values: list[float] = []
    for row in dataset.rows:
        try:
            values.append(float(row[y_idx]))
        except (TypeError, ValueError):
            return None
        labels.append(str(row[x_idx]))
    return labels, values


def _render_chart(chart: ReportChart, dataset: NormalizedToolResult) -> Drawing | None:
    extracted = _extract_xy(dataset, chart)
    if not extracted:
        return None
    labels, values = extracted

    drawing = Drawing(_CHART_WIDTH, _CHART_HEIGHT)

    if chart.type == "bar":
        bc = VerticalBarChart()
        bc.x, bc.y = 50, 30
        bc.width, bc.height = _CHART_WIDTH - 80, _CHART_HEIGHT - 60
        bc.data = [values]
        bc.categoryAxis.categoryNames = labels
        bc.categoryAxis.labels.angle = 30
        bc.bars[0].fillColor = colors.HexColor("#2980b9")
        drawing.add(bc)
    elif chart.type == "line":
        lc = HorizontalLineChart()
        lc.x, lc.y = 50, 30
        lc.width, lc.height = _CHART_WIDTH - 80, _CHART_HEIGHT - 60
        lc.data = [values]
        lc.categoryAxis.categoryNames = labels
        lc.lines[0].strokeColor = colors.HexColor("#2980b9")
        drawing.add(lc)
    elif chart.type == "pie":
        pie = Pie()
        pie.x, pie.y = _CHART_WIDTH / 2 - 75, 20
        pie.width, pie.height = 150, 150
        pie.data = values
        pie.labels = labels
        drawing.add(pie)
    else:
        return None

    return drawing


def _build_charts(spec: ReportSpecification, datasets: list[NormalizedToolResult]) -> list:
    if not spec.charts:
        return []
    flowables = [Paragraph("Charts", _STYLES["Heading2"])]
    for chart in spec.charts:
        flowables.append(Paragraph(chart.title, _STYLES["Heading3"]))
        dataset = _find_dataset(chart, datasets)
        drawing = _render_chart(chart, dataset) if dataset else None
        if drawing is not None:
            flowables.append(drawing)
        else:
            flowables.append(
                Paragraph(
                    "(Chart could not be rendered — see the corresponding "
                    "data table above.)",
                    _STYLES["Italic"],
                )
            )
        flowables.append(Spacer(1, 0.2 * inch))
    return flowables


def build_flowables(result: ReportResult) -> list:
    """Assembles the full flowable list for one report — exposed
    separately from generate_report_pdf so tests (and any future caller
    that embeds a report into a larger document) can inspect the built
    content without writing a PDF to disk."""
    spec = result.specification
    flowables: list = []
    flowables += _build_header(spec)
    flowables += _build_summary(spec)
    flowables += _build_key_findings(spec)
    flowables += _build_metrics_table(spec)
    flowables += _build_charts(spec, result.datasets)
    flowables += _build_data_tables(spec)
    return flowables


def generate_report_pdf(result: ReportResult, output_path: str | Path) -> Path:
    """Renders a ReportResult to a PDF file at output_path. Returns the
    Path for convenience. This is the only function in this module that
    touches the filesystem."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        title=result.specification.title,
    )
    flowables = build_flowables(result)
    doc.build(flowables)
    logger.info("Generated report PDF at %s", output_path)
    return output_path