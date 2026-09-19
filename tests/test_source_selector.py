from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.source_selector import source_selector
from app.llm.groq_client import reset_groq_client
from app.sources.models import DataSourceCreate, DataSourceUpdate, ProviderType


@pytest.fixture
def two_sources(isolated_env, monkeypatch):
    """Registers two Snowflake sources with distinguishing metadata (the
    canonical "Sales Snowflake" vs "Finance Snowflake" scenario from the
    product spec) and adds both to the current user's profile."""
    from app.dependencies import (
        get_current_user_profile,
        get_source_repository,
        get_user_profile_repository,
    )

    monkeypatch.setenv("DEFAULT_USER_DISPLAY_NAME", "Mike")
    from app.config import reload_settings

    reload_settings()

    repo = get_source_repository()
    sales = repo.create_source(
        DataSourceCreate(
            display_name="Sales Snowflake",
            provider=ProviderType.SNOWFLAKE,
            mcp_url="https://example.com/sales",
            pat="pat-sales",
            description="Contains CRM opportunities and sales activity",
            business_domain="sales",
        )
    )
    finance = repo.create_source(
        DataSourceCreate(
            display_name="Finance Snowflake",
            provider=ProviderType.SNOWFLAKE,
            mcp_url="https://example.com/finance",
            pat="pat-finance",
            description="Contains accounting and financial reporting data",
            business_domain="finance",
        )
    )

    profile_repo = get_user_profile_repository()
    profile = get_current_user_profile()
    profile_repo.add_source(profile, sales.id)
    profile_repo.add_source(profile, finance.id)

    return sales, finance


def _state_with_message(text: str) -> dict:
    return {
        "session_id": "s1",
        "messages": [HumanMessage(content=text)],
        "user_display_name": "Mike",
        "user_timezone": "UTC",
        "preferred_language": "en",
        "active_source_ids": [],
        "summary": "",
        "last_response": None,
    }


@pytest.fixture
def groq_env(isolated_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    from app.config import reload_settings

    reload_settings()
    reset_groq_client()
    yield
    reset_groq_client()


def test_no_sources_configured_returns_empty(isolated_env, groq_env):
    result = source_selector(_state_with_message("What was our revenue last year?"))
    assert result["active_source_ids"] == []


def test_greeting_selects_no_source(two_sources, groq_env):
    with patch("app.llm.groq_client.Groq") as MockGroq:
        fake_response = MagicMock()
        fake_response.choices = [
            MagicMock(message=MagicMock(content='{"source_ids": []}'))
        ]
        MockGroq.return_value.chat.completions.create.return_value = fake_response

        result = source_selector(_state_with_message("Hello!"))
        assert result["active_source_ids"] == []


def test_single_source_selection(two_sources, groq_env):
    sales, _finance = two_sources
    with patch("app.llm.groq_client.Groq") as MockGroq:
        fake_response = MagicMock()
        fake_response.choices = [
            MagicMock(
                message=MagicMock(content=json.dumps({"source_ids": [sales.id]}))
            )
        ]
        MockGroq.return_value.chat.completions.create.return_value = fake_response

        result = source_selector(
            _state_with_message("What was our Snowflake sales revenue last year?")
        )
        assert result["active_source_ids"] == [sales.id]


def test_multi_source_selection(two_sources, groq_env):
    sales, finance = two_sources
    with patch("app.llm.groq_client.Groq") as MockGroq:
        fake_response = MagicMock()
        fake_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({"source_ids": [sales.id, finance.id]})
                )
            )
        ]
        MockGroq.return_value.chat.completions.create.return_value = fake_response

        result = source_selector(
            _state_with_message("Compare sales revenue with our financial reports")
        )
        assert set(result["active_source_ids"]) == {sales.id, finance.id}


def test_hallucinated_id_is_filtered_out(two_sources, groq_env):
    sales, _finance = two_sources
    with patch("app.llm.groq_client.Groq") as MockGroq:
        fake_response = MagicMock()
        fake_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {"source_ids": [sales.id, "made-up-source-id"]}
                    )
                )
            )
        ]
        MockGroq.return_value.chat.completions.create.return_value = fake_response

        result = source_selector(_state_with_message("What about sales?"))
        assert result["active_source_ids"] == [sales.id]


def test_malformed_groq_response_falls_back_to_empty(two_sources, groq_env):
    with patch("app.llm.groq_client.Groq") as MockGroq:
        fake_response = MagicMock()
        fake_response.choices = [
            MagicMock(message=MagicMock(content="not valid json at all"))
        ]
        MockGroq.return_value.chat.completions.create.return_value = fake_response

        result = source_selector(_state_with_message("What about sales?"))
        assert result["active_source_ids"] == []


def test_disabled_source_excluded_from_candidates(two_sources, groq_env):
    sales, finance = two_sources
    from app.dependencies import get_source_repository

    repo = get_source_repository()
    repo.update_source(finance.id, DataSourceUpdate(enabled=False))

    with patch("app.llm.groq_client.Groq") as MockGroq:
        fake_response = MagicMock()
        MockGroq.return_value.chat.completions.create.return_value = fake_response

        source_selector(_state_with_message("Compare sales and finance"))

        # Inspect what was actually sent to Groq: disabled source must not
        # appear among the candidates offered to the LLM at all.
        call_kwargs = MockGroq.return_value.chat.completions.create.call_args.kwargs
        user_msg = call_kwargs["messages"][1]["content"]
        assert "Finance Snowflake" not in user_msg
        assert "Sales Snowflake" in user_msg