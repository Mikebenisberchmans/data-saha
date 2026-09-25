from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.llm.groq_client import reset_groq_client
from app.mcp.models import ToolCallResult, ToolInfo
from app.reports.specification import generate_report
from app.sources.models import DataSourceCreate, ProviderType


def test_generate_report_no_sources_configured(isolated_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    from app.config import reload_settings

    reload_settings()
    reset_groq_client()

    with patch("app.llm.groq_client.Groq"):
        result = generate_report("Write a report on sales performance")
        assert result.specification.title == "No data available"
        assert result.datasets == []

    reset_groq_client()


def _setup_source(monkeypatch):
    from app.dependencies import (
        get_current_user_profile,
        get_source_repository,
        get_user_profile_repository,
    )

    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    from app.config import reload_settings

    reload_settings()
    reset_groq_client()

    repo = get_source_repository()
    source = repo.create_source(
        DataSourceCreate(
            display_name="Sales Snowflake",
            provider=ProviderType.SNOWFLAKE,
            mcp_url="https://example.com/sales",
            pat="fake-pat",
            description="sales data",
            business_domain="sales",
        )
    )
    profile_repo = get_user_profile_repository()
    profile = get_current_user_profile()
    profile_repo.add_source(profile, source.id)
    return source


def _gather_mock(source, columns, rows):
    fake_tool = ToolInfo(source_id=source.id, tool_name="execute_sql", description="")

    def _fake_create(**kwargs):
        system_content = kwargs["messages"][0]["content"]
        response = MagicMock()
        if "routing component" in system_content:
            response.choices = [
                MagicMock(
                    message=MagicMock(
                        content=json.dumps({"source_ids": [source.id]}), tool_calls=None
                    )
                )
            ]
            return response

        tools = kwargs.get("tools")
        if tools:
            already_called = any(m.get("role") == "tool" for m in kwargs["messages"])
            if not already_called:
                tc = MagicMock()
                tc.id = "call_1"
                tc.function.name = f"{source.id}__execute_sql"
                tc.function.arguments = json.dumps({"query": "SELECT * FROM sales"})
                response.choices = [
                    MagicMock(message=MagicMock(content=None, tool_calls=[tc]))
                ]
            else:
                response.choices = [
                    MagicMock(message=MagicMock(content="", tool_calls=None))
                ]
            return response

        return None  # caller fills in the narrative response

    return fake_tool, _fake_create


def test_generate_report_deterministic_fields_computed_from_real_data(
    isolated_env, monkeypatch
):
    """data_sources, query_information, tables, and metrics must be
    correct even if the LLM's narrative call fails entirely."""
    source = _setup_source(monkeypatch)
    fake_tool, _fake_create_base = _gather_mock(
        source, ["region", "revenue"], [["US", 1000], ["EU", 500]]
    )

    def _fake_create(**kwargs):
        result = _fake_create_base(**kwargs)
        if result is not None:
            return result
        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(content="not json", tool_calls=None))
        ]
        return response

    with patch("app.analytics.gathering.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock(
            return_value=ToolCallResult(
                source_id=source.id,
                tool_name="execute_sql",
                content=json.dumps(
                    {"columns": ["region", "revenue"], "rows": [["US", 1000], ["EU", 500]]}
                ),
            )
        )
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

        MockGroq.return_value.chat.completions.create.side_effect = _fake_create

        result = generate_report("Write a report on sales revenue by region")

    spec = result.specification
    assert spec.data_sources == ["Sales Snowflake"]
    assert len(spec.query_information) == 1
    assert spec.query_information[0].source_id == source.id
    assert spec.query_information[0].tool_name == "execute_sql"
    assert len(spec.tables) == 1
    assert spec.tables[0].columns == ["region", "revenue"]
    assert spec.tables[0].rows == [["US", 1000], ["EU", 500]]
    assert len(spec.metrics) == 1
    assert spec.metrics[0].value == 1500.0
    assert spec.generation_timestamp is not None
    assert "narrative summary could not be generated" in spec.summary

    reset_groq_client()


def test_generate_report_full_happy_path(isolated_env, monkeypatch):
    source = _setup_source(monkeypatch)
    fake_tool, _fake_create_base = _gather_mock(
        source, ["region", "revenue"], [["US", 1000], ["EU", 500]]
    )

    narrative = {
        "title": "Q4 Sales Report",
        "summary": "Sales performed well across regions.",
        "key_findings": ["US led with $1000 in revenue.", "EU contributed $500."],
        "charts": [
            {"type": "bar", "title": "Revenue by Region", "x": "region", "y": "revenue", "source_id": source.id}
        ],
    }

    def _fake_create(**kwargs):
        result = _fake_create_base(**kwargs)
        if result is not None:
            return result
        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(narrative), tool_calls=None))
        ]
        return response

    with patch("app.analytics.gathering.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock(
            return_value=ToolCallResult(
                source_id=source.id,
                tool_name="execute_sql",
                content=json.dumps(
                    {"columns": ["region", "revenue"], "rows": [["US", 1000], ["EU", 500]]}
                ),
            )
        )
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

        MockGroq.return_value.chat.completions.create.side_effect = _fake_create

        result = generate_report("Write a report on sales revenue by region")

    spec = result.specification
    assert spec.title == "Q4 Sales Report"
    assert spec.summary == "Sales performed well across regions."
    assert len(spec.key_findings) == 2
    assert len(spec.charts) == 1
    assert spec.charts[0].x == "region"
    assert spec.data_sources == ["Sales Snowflake"]
    assert len(spec.tables) == 1
    assert len(result.datasets) == 1

    reset_groq_client()