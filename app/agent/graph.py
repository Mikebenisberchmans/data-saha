"""
LangGraph graph assembly.

Phase 2 graph is intentionally minimal:

    START -> load_context -> conversation -> END

Source selection, tool execution, analysis, and the dashboard/report
branches are added in later phases per the implementation roadmap; the
node functions already take/return the full AgentState so wiring them in
later is a graph-shape change, not a state-shape change.

Multi-turn memory: a LangGraph `MemorySaver` checkpointer is used, keyed by
`thread_id` (= our session_id), so the graph remembers prior turns within
this process's lifetime. This is intentionally NOT the durable, summarized
conversation memory described in the product spec (section 12-13) — that
is Phase 3's job (disk persistence + summarizer node). Phase 3 will likely
replace MemorySaver with a custom checkpointer or an explicit
load/save-around-invoke pattern; nothing in this graph's shape depends on
which one is used.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes.context import load_context
from app.agent.nodes.conversation import conversation
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context)
    graph.add_node("conversation", conversation)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "conversation")
    graph.add_edge("conversation", END)

    return graph.compile(checkpointer=MemorySaver())


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def reset_graph() -> None:
    """Test helper — forces the graph (and its checkpointer) to be rebuilt."""
    global _compiled_graph
    _compiled_graph = None