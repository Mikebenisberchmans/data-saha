from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.llm.groq_client import reset_groq_client
from app.memory.summarizer import estimate_tokens, maybe_summarize, summarize


@pytest.fixture
def mocked_groq(isolated_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    from app.config import reload_settings

    reload_settings()
    reset_groq_client()

    with patch("app.llm.groq_client.Groq") as MockGroq:
        instance = MockGroq.return_value

        def _fake_create(**kwargs):
            response = MagicMock()
            response.choices = [
                MagicMock(message=MagicMock(content="updated summary text"))
            ]
            return response

        instance.chat.completions.create.side_effect = _fake_create
        yield instance

    reset_groq_client()


def test_estimate_tokens_rough_heuristic():
    messages = [HumanMessage(content="a" * 400)]  # ~100 tokens at 4 chars/token
    assert estimate_tokens(messages) == 100


def test_summarize_returns_previous_summary_if_nothing_to_drop(mocked_groq):
    result = summarize("old summary", [])
    assert result == "old summary"
    mocked_groq.chat.completions.create.assert_not_called()


def test_summarize_calls_groq_and_returns_new_summary(mocked_groq):
    result = summarize("old summary", [HumanMessage(content="some old message")])
    assert result == "updated summary text"
    mocked_groq.chat.completions.create.assert_called_once()


def test_maybe_summarize_below_threshold_no_op(isolated_env, monkeypatch):
    monkeypatch.setenv("SUMMARY_TRIGGER_TOKENS", "1000000")
    from app.config import reload_settings

    reload_settings()

    messages = [HumanMessage(content="short"), AIMessage(content="also short")]
    result_messages, result_summary = maybe_summarize(messages, "")
    assert result_messages == messages
    assert result_summary == ""


def test_maybe_summarize_above_threshold_trims_and_summarizes(mocked_groq, monkeypatch):
    monkeypatch.setenv("SUMMARY_TRIGGER_TOKENS", "10")  # trivially low
    monkeypatch.setenv("RECENT_MESSAGES_KEEP", "2")
    from app.config import reload_settings

    reload_settings()

    messages = [
        HumanMessage(content="message one is fairly long here"),
        AIMessage(content="reply one is also fairly long here"),
        HumanMessage(content="message two"),
        AIMessage(content="reply two"),
    ]

    result_messages, result_summary = maybe_summarize(messages, "")

    assert len(result_messages) == 2  # RECENT_MESSAGES_KEEP
    assert result_messages == messages[-2:]
    assert result_summary == "updated summary text"
    mocked_groq.chat.completions.create.assert_called_once()


def test_maybe_summarize_skips_trim_if_not_enough_messages(mocked_groq, monkeypatch):
    monkeypatch.setenv("SUMMARY_TRIGGER_TOKENS", "1")
    monkeypatch.setenv("RECENT_MESSAGES_KEEP", "10")
    from app.config import reload_settings

    reload_settings()

    messages = [HumanMessage(content="only one message")]
    result_messages, result_summary = maybe_summarize(messages, "")

    # Only 1 message but keep_n=10 -> nothing to drop, no-op
    assert result_messages == messages
    assert result_summary == ""
    mocked_groq.chat.completions.create.assert_not_called()