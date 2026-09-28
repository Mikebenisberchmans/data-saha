"""
POST /report - delegates to app.reports.specification.generate_report
(Phase 10). Returns the structured ReportSpecification as JSON by default;
pass format="pdf" to get a rendered PDF file back instead (via
app.reports.generator.generate_report_pdf), streamed back as the response
body with the correct content type.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.reports.generator import generate_report_pdf
from app.reports.specification import ReportResult, generate_report

router = APIRouter(tags=["report"])


class ReportRequest(BaseModel):
    question: str
    format: Literal["json", "pdf"] = "json"


@router.post("/report")
def report(request: ReportRequest):
    result = generate_report(request.question)

    if request.format == "json":
        return result

    # PDF: render to a temp file and stream it back. The temp directory
    # persists for the process lifetime, which is fine for a local desktop
    # backend serving one user; a hosted multi-user deployment would want
    # explicit cleanup here instead.
    tmp_dir = Path(tempfile.gettempdir()) / "data-saha-reports"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c if c.isalnum() else "_" for c in result.specification.title)[:60]
    output_path = tmp_dir / f"{safe_title or 'report'}.pdf"
    generate_report_pdf(result, output_path)

    return FileResponse(
        path=output_path,
        media_type="application/pdf",
        filename=output_path.name,
    )