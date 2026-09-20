r"""
LangGraph graph assembly.

Phase 6 graph:

    START -> load_context -> source_selector --+-- no source --> conversation --+
                                                 \-- source(s) --> tool_executor -+--> summarizer -> END

tool_executor (new in Phase 6) runs the full agentic tool-calling loop
against the source(s) source_selector picked: it discovers each source's
tools (Phase 4), lets Groq call them (possibly more than one, possibly
more than once — per product spec section 15, this is NOT a single-shot
LLM->tool->answer pipeline), and produces the final answer itself. Turns
with no relevant source still go through the plain `conversation` node,
unchanged from Phase 5.

Conversation persistence across process runs is handled explicitly by
ConversationStore + app/agent/runner.py, not by a LangGraph checkpointer
(see Phase 3 notes in app/agent/state.py for why).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.context import load_context
from app.agent.nodes.conversation import conversation
from app.agent.nodes.source_selector import source_selector
from app.agent.nodes.summarizer import summarizer
from app.agent.nodes.tool_executor import tool_executor
from app.agent.state import AgentState


def _route_after_source_selection(state: AgentState) -> str:
    return "tool_executor" if state.get("active_source_ids") else "conversation"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context)
    graph.add_node("source_selector", source_selector)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("conversation", conversation)
    graph.add_node("summarizer", summarizer)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "source_selector")
    graph.add_conditional_edges(
        "source_selector",
        _route_after_source_selection,
        {"tool_executor": "tool_executor", "conversation": "conversation"},
    )
    graph.add_edge("tool_executor", "summarizer")
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