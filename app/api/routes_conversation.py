"""
GET /conversation/{session_id} - returns the persisted conversation record
(messages, rolling summary, active sources) for a session. Read-only;
creation/mutation happens only via POST /chat.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import get_conversation_store
from app.memory.conversation import ConversationRecord

router = APIRouter(tags=["conversation"])


@router.get("/conversation/{session_id}", response_model=ConversationRecord)
def get_conversation(session_id: str) -> ConversationRecord:
    store = get_conversation_store()
    return store.load(session_id)