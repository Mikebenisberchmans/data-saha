"""
Groq LLM client wrapper.

Model name and API key are never hard-coded — always read from settings
(app/config.py), which in turn reads GROQ_API_KEY / GROQ_MODEL from the
environment. This client is intentionally thin and SDK-shape-agnostic:
LangGraph nodes call it and get back the raw Groq response; it does not
know about LangGraph, sources, or MCP.
"""

from __future__ import annotations

from typing import Any

from groq import Groq

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqClientError(RuntimeError):
    pass


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        settings = get_settings()
        self._api_key = api_key or settings.groq_api_key
        self._model = model or settings.groq_model

        if not self._api_key:
            raise GroqClientError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        if not self._model:
            raise GroqClientError(
                "GROQ_MODEL is not set. Add it to your .env file, e.g. "
                "GROQ_MODEL=llama-3.3-70b-versatile"
            )

        self._client = Groq(api_key=self._api_key)

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.2,
    ):
        """Thin wrapper around chat.completions.create. Returns the raw Groq
        response object; callers extract what they need (message content,
        tool_calls, etc). `tools` is accepted now so the signature doesn't
        need to change in Phase 6 (MCP tool calling) — it's simply unused
        until then."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # Groq SDK raises various API error types
            logger.error("Groq API call failed: %s", type(exc).__name__)
            raise GroqClientError(f"Groq API call failed: {exc}") from exc


_client: GroqClient | None = None


def get_groq_client() -> GroqClient:
    global _client
    if _client is None:
        _client = GroqClient()
    return _client


def reset_groq_client() -> None:
    """Test helper — forces the singleton to be rebuilt on next access."""
    global _client
    _client = None