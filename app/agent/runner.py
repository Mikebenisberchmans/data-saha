"""
Conversation runner.

This is the seam between "what's persisted between process runs" (a
ConversationRecord, plain dicts) and "what a single graph.invoke() call
needs" (LangChain message objects). Callers — the CLI test client now,
the FastAPI /chat route starting Phase 11 — should call `run_turn` rather
than touching graph.invoke() or the ConversationStore directly, so this
conversion and the persistence side-effect happen in exactly one place.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.dependencies import get_conversation_store
from app.memory.conversation import StoredMessage


def _stored_to_langchain(stored: list[StoredMessage]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for m in stored:
        if m.role == "assistant":
            result.append(AIMessage(content=m.content))
        else:
            result.append(HumanMessage(content=m.content))
    return result


def _langchain_to_stored(messages: list[BaseMessage]) -> list[StoredMessage]:
    return [
        StoredMessage(
            role="assistant" if isinstance(m, AIMessage) else "user",
            content=m.content,
        )
        for m in messages
    ]


def run_turn(session_id: str, user_message: str) -> str:
    """Runs one user turn against the persisted conversation for
    `session_id`, persists the updated conversation (trimmed/summarized if
    the summarizer node triggered), and returns the assistant's reply."""
    store = get_conversation_store()
    record = store.load(session_id)

    history = _stored_to_langchain(record.messages)
    history.append(HumanMessage(content=user_message))

    initial_state: AgentState = {
        "session_id": session_id,
        "messages": history,
        "user_display_name": "",
        "user_timezone": "",
        "preferred_language": "",
        "active_source_ids": record.active_source_ids,
        "summary": record.summary,
        "last_response": None,
    }

    graph = get_graph()
    result = graph.invoke(initial_state)

    record.messages = _langchain_to_stored(result["messages"])
    record.summary = result.get("summary", record.summary)
    record.active_source_ids = result.get(
        "active_source_ids", record.active_source_ids
    )
    store.save(record)

    return result["last_response"] or ""