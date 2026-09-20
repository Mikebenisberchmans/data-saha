from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agent.graph import reset_graph
from app.agent.runner import run_turn
from app.llm.groq_client import reset_groq_client


@pytest.fixture
def mocked_groq(isolated_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("DEFAULT_USER_DISPLAY_NAME", "Mike")
    from app.config import reload_settings

    reload_settings()
    reset_groq_client()
    reset_graph()

    with patch("app.llm.groq_client.Groq") as MockGroq:
        instance = MockGroq.return_value

        def _fake_create(**kwargs):
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="mocked reply"))]
            return response

        instance.chat.completions.create.side_effect = _fake_create
        yield instance

    reset_groq_client()
    reset_graph()


def test_run_turn_returns_reply(mocked_groq):
    reply = run_turn("session-1", "hello")
    assert reply == "mocked reply"


def test_run_turn_persists_across_simulated_restart(mocked_groq):
    from app.dependencies import get_conversation_store

    run_turn("session-2", "first message")
    get_conversation_store.cache_clear()
    run_turn("session-2", "second message")

    store = get_conversation_store()
    record = store.load("session-2")
    assert len(record.messages) == 4
    assert record.messages[0].content == "first message"
    assert record.messages[2].content == "second message"


def test_different_sessions_do_not_share_history(mocked_groq):
    from app.dependencies import get_conversation_store

    run_turn("session-a", "hi from a")
    run_turn("session-b", "hi from b")

    store = get_conversation_store()
    record_a = store.load("session-a")
    record_b = store.load("session-b")

    assert len(record_a.messages) == 2
    assert len(record_b.messages) == 2
    assert record_a.messages[0].content == "hi from a"
    assert record_b.messages[0].content == "hi from b"


def test_run_turn_triggers_summarization_when_threshold_exceeded(
    mocked_groq, monkeypatch
):
    from app.dependencies import get_conversation_store

    monkeypatch.setenv("SUMMARY_TRIGGER_TOKENS", "5")
    monkeypatch.setenv("RECENT_MESSAGES_KEEP", "2")
    from app.config import reload_settings

    reload_settings()

    run_turn("session-3", "first message is long enough to matter")
    run_turn("session-3", "second message")

    store = get_conversation_store()
    record = store.load("session-3")

    assert len(record.messages) <= 2
    assert record.summary != ""


def test_run_turn_persists_selected_source_ids(mocked_groq):
    """End-to-end: source_selector picks a source, conversation node still
    runs and replies, and the selection is persisted in the conversation
    record's active_source_ids (Phase 5's contribution to the runner)."""
    from app.dependencies import (
        get_conversation_store,
        get_current_user_profile,
        get_source_repository,
        get_user_profile_repository,
    )
    from app.sources.models import DataSourceCreate, ProviderType

    repo = get_source_repository()
    source = repo.create_source(
        DataSourceCreate(
            display_name="Sales Snowflake",
            provider=ProviderType.SNOWFLAKE,
            mcp_url="https://example.com/sales",
            pat="fake-pat",
            description="CRM and sales data",
            business_domain="sales",
        )
    )
    profile_repo = get_user_profile_repository()
    profile = get_current_user_profile()
    profile_repo.add_source(profile, source.id)

    # The single mocked `_fake_create` in this fixture returns
    # "mocked reply" for *every* Groq call, including the source_selector's
    # JSON-expecting call. That's not valid JSON, so source_selector will
    # safely fall back to an empty selection. This test is about the
    # runner→persistence wiring, not the selector's LLM output.
    reply = run_turn("session-4", "What was our Snowflake sales revenue?")
    assert reply == "mocked reply"

    store = get_conversation_store()
    record = store.load("session-4")
    assert record.active_source_ids == []  # malformed selector output -> []


def test_run_turn_with_realistic_selector_response_persists_source_id(
    isolated_env, monkeypatch
):
    """Confirms source selection still persists correctly now that a
    selected source routes to tool_executor (Phase 6) instead of
    conversation (Phase 5). Tool discovery itself is mocked here — the
    real, unmocked MCP round trip is covered separately by
    test_run_turn_answers_using_real_mcp_tool_data below."""
    import json
    from unittest.mock import AsyncMock

    from app.agent.graph import reset_graph
    from app.dependencies import (
        get_conversation_store,
        get_current_user_profile,
        get_source_repository,
        get_user_profile_repository,
    )
    from app.llm.groq_client import reset_groq_client
    from app.sources.models import DataSourceCreate, ProviderType

    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("DEFAULT_USER_DISPLAY_NAME", "Mike")
    from app.config import reload_settings

    reload_settings()
    reset_groq_client()
    reset_graph()

    repo = get_source_repository()
    source = repo.create_source(
        DataSourceCreate(
            display_name="Sales Snowflake",
            provider=ProviderType.SNOWFLAKE,
            mcp_url="https://example.com/sales",
            pat="fake-pat",
            description="CRM and sales data",
            business_domain="sales",
        )
    )
    profile_repo = get_user_profile_repository()
    profile = get_current_user_profile()
    profile_repo.add_source(profile, source.id)

    def _fake_create(**kwargs):
        system_content = kwargs["messages"][0]["content"]
        response = MagicMock()
        if "routing component" in system_content:
            content = json.dumps({"source_ids": [source.id]})
        else:
            content = "unused in this test"
        response.choices = [
            MagicMock(message=MagicMock(content=content, tool_calls=None))
        ]
        return response

    with patch("app.llm.groq_client.Groq") as MockGroq, \
         patch("app.agent.nodes.tool_executor.get_mcp_manager") as mock_get_manager:
        MockGroq.return_value.chat.completions.create.side_effect = _fake_create

        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[])  # no real connection
        mock_get_manager.return_value = manager

        reply = run_turn("session-5", "What was our Snowflake sales revenue?")

    # No tools discovered -> tool_executor's deterministic no-Groq-call
    # fallback fires.
    assert "couldn't reach any usable tools" in reply

    store = get_conversation_store()
    record = store.load("session-5")
    assert record.active_source_ids == [source.id]

    reset_groq_client()
    reset_graph()


def _start_local_mcp_server(port: int):
    import threading
    import time

    from mcp.server.mcpserver import MCPServer

    server = MCPServer("test-analytics-server")

    @server.tool()
    def execute_sql(query: str) -> dict:
        """Execute a read-only SQL query against the test warehouse."""
        return {"columns": ["region", "revenue"], "rows": [["US", 4200000]]}

    def run():
        server.run(transport="streamable-http", host="127.0.0.1", port=port)

    threading.Thread(target=run, daemon=True).start()
    time.sleep(1.5)


def test_run_turn_answers_using_real_mcp_tool_data(isolated_env, monkeypatch):
    """The big one: a real local MCP server (no mocking of the MCP SDK at
    all), a real SourceRepository-registered source, a real MCPManager
    connect->discover->call round trip, driven through the ACTUAL graph via
    run_turn — with only Groq's decisions mocked (no live Groq key in
    CI/tests). Proves the full Phase 4+5+6 stack actually produces an
    answer grounded in live tool data, not just that the pieces work in
    isolation.

    NOTE: this is intentionally a plain sync test, not `async def` with
    `@pytest.mark.asyncio` — run_turn() is sync and drives its own asyncio
    event loop internally (see tool_executor.py). Calling it from inside
    pytest-asyncio's already-running loop would raise "asyncio.run()
    cannot be called from a running event loop"."""
    import json

    from app.agent.graph import reset_graph
    from app.dependencies import (
        get_conversation_store,
        get_current_user_profile,
        get_source_repository,
        get_user_profile_repository,
    )
    from app.llm.groq_client import reset_groq_client
    from app.sources.models import DataSourceCreate, ProviderType

    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("DEFAULT_USER_DISPLAY_NAME", "Mike")
    from app.config import reload_settings

    reload_settings()
    reset_groq_client()
    reset_graph()

    port = 8770
    _start_local_mcp_server(port)

    repo = get_source_repository()
    source = repo.create_source(
        DataSourceCreate(
            display_name="Sales Snowflake",
            provider=ProviderType.SNOWFLAKE,
            mcp_url=f"http://127.0.0.1:{port}/mcp",
            pat="fake-pat-not-checked-by-test-server",
            description="Contains CRM opportunities and sales revenue",
            business_domain="sales",
        )
    )
    profile_repo = get_user_profile_repository()
    profile = get_current_user_profile()
    profile_repo.add_source(profile, source.id)

    def _fake_create(**kwargs):
        system_content = kwargs["messages"][0]["content"]
        response = MagicMock()

        if "routing component" in system_content:
            response.choices = [
                MagicMock(
                    message=MagicMock(
                        content=json.dumps({"source_ids": [source.id]}),
                        tool_calls=None,
                    )
                )
            ]
            return response

        tools = kwargs.get("tools")
        already_called_tool = any(m.get("role") == "tool" for m in kwargs["messages"])

        if tools and not already_called_tool:
            tool_call = MagicMock()
            tool_call.id = "call_1"
            tool_call.function.name = f"{source.id}__execute_sql"
            tool_call.function.arguments = json.dumps({"query": "SELECT revenue"})
            response.choices = [
                MagicMock(message=MagicMock(content=None, tool_calls=[tool_call]))
            ]
        else:
            tool_messages = [m for m in kwargs["messages"] if m.get("role") == "tool"]
            assert tool_messages, "expected the tool result to be in context by now"
            assert "4200000" in tool_messages[0]["content"]
            response.choices = [
                MagicMock(
                    message=MagicMock(
                        content="Your sales revenue was $4,200,000 (US region).",
                        tool_calls=None,
                    )
                )
            ]
        return response

    with patch("app.llm.groq_client.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.side_effect = _fake_create

        reply = run_turn("session-mcp", "What was our Snowflake sales revenue?")

    assert "4,200,000" in reply

    store = get_conversation_store()
    record = store.load("session-mcp")
    assert record.active_source_ids == [source.id]

    reset_groq_client()
    reset_graph()