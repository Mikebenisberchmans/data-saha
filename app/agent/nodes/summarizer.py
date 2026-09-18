"""
summarizer node: the last node in the graph. Applies the summarization
trigger/trim logic (app/memory/summarizer.py) to the current turn's full
message list before the runner persists it.
"""

from __future__ import annotations

from app.agent.state import AgentState
from app.memory.summarizer import maybe_summarize


def summarizer(state: AgentState) -> dict:
    trimmed_messages, updated_summary = maybe_summarize(
        state["messages"], state.get("summary", "")
    )
    return {"messages": trimmed_messages, "summary": updated_summary}