"""
POST /chat — the main conversational endpoint. Delegates to
app.agent.runner.run_turn (built across Phases 2-8), which already handles
persistence, source selection, MCP tool execution, SQL safety, and
summarization internally. This route is a thin HTTP adapter over it — no
business logic lives here.

run_turn() is a synchronous function that drives its own asyncio event
loop internally (see app/agent/nodes/tool_executor.py's docstring for why).
Route handlers here are therefore plain `def`, not `async def` — FastAPI
runs sync path operations in a thread pool automatically, which is exactly
what's needed to avoid nesting asyncio.run() inside the server's own event
loop.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.runner import run_turn
from app.core.logging import get_logger
from app.dependencies import get_conversation_store, get_default_session_id
from app.llm.groq_client import GroqClientError

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    dashboard_available: bool
    report_available: bool
    sources_used: list[str]


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or get_default_session_id()

    try:
        reply = run_turn(session_id, request.message)
    except GroqClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    store = get_conversation_store()
    record = store.load(session_id)
    sources_used = record.active_source_ids

    return ChatResponse(
        session_id=session_id,
        message=reply,
        # A dashboard/report can meaningfully be built from this turn's
        # context only if it actually drew on live source data.
        dashboard_available=bool(sources_used),
        report_available=bool(sources_used),
        sources_used=sources_used,
    )