from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.dashboard.specification import (
    ChartSpec,
    DashboardSpecification,
    generate_dashboard,
)
from app.llm.groq_client import reset_groq_client
from app.mcp.models import ToolCallResult, ToolInfo
from app.sources.models import DataSourceCreate, ProviderType


def test_chart_spec_rejects_unknown_type():
    with pytest.raises(Exception):
        ChartSpec(type="pie3d", title="bad")


def test_dashboard_specification_defaults():
    spec = DashboardSpecification(title="Sales Overview")
    assert spec.description == ""
    assert spec.charts == []
    assert spec.metrics == []
    assert spec.filters == []


def test_dashboard_specification_full_roundtrip():
    spec = DashboardSpecification(
        title="Year-over-Year Sales Performance",
        description="Revenue trends",
        charts=[
            ChartSpec(type="line", title="Revenue by Year", x="year", y="revenue", source_id="src-a"),
            ChartSpec(type="bar", title="Revenue by Region", x="region", y="revenue", source_id="src-a"),
        ],
        metrics=[{"name": "YoY Growth", "value": 14.3, "unit": "percent"}],
        filters=["region", "product"],
    )
    dumped = spec.model_dump()
    assert dumped["charts"][0]["type"] == "line"
    assert dumped["metrics"][0]["unit"] == "percent"


def test_generate_dashboard_no_sources_configured(isolated_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    from app.config import reload_settings

    reload_settings()
    reset_groq_client()

    with patch("app.llm.groq_client.Groq") as MockGroq:
        result = generate_dashboard("Show me sales performance")
        assert result.specification.title == "No data available"
        assert result.datasets == []

    reset_groq_client()


def test_generate_dashboard_malformed_spec_json_falls_back_gracefully(
    isolated_env, monkeypatch
):
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

    fake_tool = ToolInfo(source_id=source.id, tool_name="execute_sql", description="")

    with patch("app.analytics.gathering.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock(
            return_value=ToolCallResult(
                source_id=source.id,
                tool_name="execute_sql",
                content=json.dumps(
                    {"columns": ["region", "revenue"], "rows": [["US", 1000]]}
                ),
            )
        )
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

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
            if tools:
                already_called = any(
                    m.get("role") == "tool" for m in kwargs["messages"]
                )
                if not already_called:
                    tc = MagicMock()
                    tc.id = "call_1"
                    tc.function.name = f"{source.id}__execute_sql"
                    tc.function.arguments = json.dumps({"query": "SELECT revenue FROM sales"})
                    response.choices = [
                        MagicMock(message=MagicMock(content=None, tool_calls=[tc]))
                    ]
                else:
                    response.choices = [
                        MagicMock(message=MagicMock(content="", tool_calls=None))
                    ]
                return response

            response.choices = [
                MagicMock(message=MagicMock(content="not valid json", tool_calls=None))
            ]
            return response

        MockGroq.return_value.chat.completions.create.side_effect = _fake_create

        result = generate_dashboard("Show me sales revenue by region")

    assert result.specification.title == "Dashboard generation failed"
    assert len(result.datasets) == 1
    assert result.datasets[0].source_id == source.id
    assert result.datasets[0].columns == ["region", "revenue"]

    reset_groq_client()


def test_generate_dashboard_full_happy_path(isolated_env, monkeypatch):
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

    fake_tool = ToolInfo(source_id=source.id, tool_name="execute_sql", description="")

    valid_spec = {
        "title": "Sales by Region",
        "description": "Revenue broken down by region",
        "charts": [
            {"type": "bar", "title": "Revenue by Region", "x": "region", "y": "revenue", "source_id": source.id}
        ],
        "metrics": [{"name": "Total Revenue", "value": 1800, "unit": "USD"}],
        "filters": ["region"],
    }

    with patch("app.analytics.gathering.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock(
            return_value=ToolCallResult(
                source_id=source.id,
                tool_name="execute_sql",
                content=json.dumps(
                    {"columns": ["region", "revenue"], "rows": [["US", 1000], ["EU", 800]]}
                ),
            )
        )
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

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
            if tools:
                already_called = any(m.get("role") == "tool" for m in kwargs["messages"])
                if not already_called:
                    tc = MagicMock()
                    tc.id = "call_1"
                    tc.function.name = f"{source.id}__execute_sql"
                    tc.function.arguments = json.dumps({"query": "SELECT region, revenue FROM sales"})
                    response.choices = [
                        MagicMock(message=MagicMock(content=None, tool_calls=[tc]))
                    ]
                else:
                    response.choices = [
                        MagicMock(message=MagicMock(content="", tool_calls=None))
                    ]
                return response

            response.choices = [
                MagicMock(message=MagicMock(content=json.dumps(valid_spec), tool_calls=None))
            ]
            return response

        MockGroq.return_value.chat.completions.create.side_effect = _fake_create

        result = generate_dashboard("Show me sales revenue by region")

    assert result.specification.title == "Sales by Region"
    assert len(result.specification.charts) == 1
    assert result.specification.charts[0].x == "region"
    assert result.specification.metrics[0].value == 1800
    assert len(result.datasets) == 1
    assert result.datasets[0].columns == ["region", "revenue"]
    assert result.datasets[0].rows == [["US", 1000], ["EU", 800]]

    reset_groq_client()