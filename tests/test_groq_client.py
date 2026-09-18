from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.llm.groq_client import GroqClient, GroqClientError, reset_groq_client


@pytest.fixture(autouse=True)
def _reset_client():
    reset_groq_client()
    yield
    reset_groq_client()


def test_missing_api_key_raises():
    with pytest.raises(GroqClientError, match="GROQ_API_KEY"):
        GroqClient(api_key=None, model="openai/gpt-oss-20b")


def test_missing_model_raises():
    with pytest.raises(GroqClientError, match="GROQ_MODEL"):
        GroqClient(api_key="fake-key", model=None)


def test_chat_returns_groq_response():
    with patch("app.llm.groq_client.Groq") as MockGroq:
        instance = MockGroq.return_value
        fake_response = MagicMock()
        instance.chat.completions.create.return_value = fake_response

        client = GroqClient(api_key="fake-key", model="openai/gpt-oss-20b")
        result = client.chat(messages=[{"role": "user", "content": "hi"}])

        assert result is fake_response
        instance.chat.completions.create.assert_called_once()
        call_kwargs = instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-oss-20b"
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_wraps_sdk_errors():
    with patch("app.llm.groq_client.Groq") as MockGroq:
        instance = MockGroq.return_value
        instance.chat.completions.create.side_effect = RuntimeError("boom")

        client = GroqClient(api_key="fake-key", model="openai/gpt-oss-20b")
        with pytest.raises(GroqClientError):
            client.chat(messages=[{"role": "user", "content": "hi"}])