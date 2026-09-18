"""
LangGraph graph assembly.

Phase 3 graph:

    START -> load_context -> conversation -> summarizer -> END

Change from Phase 2: conversation persistence across process runs is now
handled explicitly by ConversationStore + app/agent/runner.py, not by a
LangGraph checkpointer. Phase 2 used an in-memory `MemorySaver`
checkpointer keyed by thread_id for short-lived multi-turn memory within a
single process; that's dropped here because (a) it didn't survive process
restarts anyway, which the product needs, and (b) the summarizer needs to
fully replace the message list (trim old messages), which doesn't map
cleanly onto checkpointer + reducer semantics. See app/agent/state.py for
the corresponding `messages` type change.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.context import load_context
from app.agent.nodes.conversation import conversation
from app.agent.nodes.summarizer import summarizer
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context)
    graph.add_node("conversation", conversation)
    graph.add_node("summarizer", summarizer)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "conversation")
    graph.add_edge("conversation", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def reset_graph() -> None:
    """Test helper — forces the graph to be rebuilt."""
    global _compiled_graph
    _compiled_graph = None