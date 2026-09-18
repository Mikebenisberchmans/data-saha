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

    # Simulate a process restart: clear the cached store so the next call
    # re-reads from disk rather than reusing an in-memory object.
    get_conversation_store.cache_clear()

    run_turn("session-2", "second message")

    store = get_conversation_store()
    record = store.load("session-2")
    # 2 user turns + 2 assistant replies = 4 stored messages
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

    monkeypatch.setenv("SUMMARY_TRIGGER_TOKENS", "5")  # trivially low
    monkeypatch.setenv("RECENT_MESSAGES_KEEP", "2")
    from app.config import reload_settings

    reload_settings()

    run_turn("session-3", "first message is long enough to matter")
    run_turn("session-3", "second message")

    store = get_conversation_store()
    record = store.load("session-3")

    assert len(record.messages) <= 2
    assert record.summary != ""