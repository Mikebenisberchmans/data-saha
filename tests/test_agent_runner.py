from __future__ import annotations

import json
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
    # safely fall back to an empty selection (see test_source_selector.py
    # for the case where a proper JSON payload is returned). This test is
    # about the runner→persistence wiring, not the selector's LLM output.
    reply = run_turn("session-4", "What was our Snowflake sales revenue?")
    assert reply == "mocked reply"

    store = get_conversation_store()
    record = store.load("session-4")
    assert record.active_source_ids == []  # malformed selector output -> []


def test_run_turn_with_realistic_selector_response_persists_source_id(
    isolated_env, monkeypatch
):
    """Same as above, but with a mock that actually distinguishes the
    source_selector's system prompt from the conversation node's, and
    returns valid selector JSON — proving a real selection flows all the
    way through run_turn into the persisted record."""
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
            content = "Here's what I found about your sales data."
        response.choices = [MagicMock(message=MagicMock(content=content))]
        return response

    with patch("app.llm.groq_client.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.side_effect = _fake_create

        reply = run_turn("session-5", "What was our Snowflake sales revenue?")

    assert "sales" in reply.lower()

    store = get_conversation_store()
    record = store.load("session-5")
    assert record.active_source_ids == [source.id]

    reset_groq_client()
    reset_graph()