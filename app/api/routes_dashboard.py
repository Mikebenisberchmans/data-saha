"""
POST /dashboard - delegates to app.dashboard.specification.generate_dashboard
(Phase 9). The backend never generates frontend code; this just exposes
that capability over HTTP so the future Tauri/React app can call it
directly, per product spec section 24 - dashboard generation is triggered
by an explicit API call, never as an implicit side effect of chat.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.dashboard.specification import DashboardResult, generate_dashboard

router = APIRouter(tags=["dashboard"])


class DashboardRequest(BaseModel):
    question: str


@router.post("/dashboard", response_model=DashboardResult)
def dashboard(request: DashboardRequest) -> DashboardResult:
    return generate_dashboard(request.question)