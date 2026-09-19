"""
LangGraph graph assembly.

Phase 5 graph:

    START -> load_context -> source_selector -> conversation -> summarizer -> END

source_selector (new in Phase 5) decides which configured data source(s),
if any, are relevant to the user's latest message, using only public
source metadata (display_name, provider, description, business_domain —
never mcp_url or credential_ref). It does not connect to MCP or call any
tool — that's Phase 6 (tool_execution), which will sit between
source_selector and conversation once built.

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
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context)
    graph.add_node("source_selector", source_selector)
    graph.add_node("conversation", conversation)
    graph.add_node("summarizer", summarizer)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "source_selector")
    graph.add_edge("source_selector", "conversation")
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