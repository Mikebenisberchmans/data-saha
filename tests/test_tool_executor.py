from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.tool_executor import (
    MAX_TOOL_ITERATIONS,
    _namespaced_name,
    _split_namespaced_name,
    tool_executor,
)
from app.llm.groq_client import reset_groq_client
from app.mcp.models import ToolCallResult, ToolInfo
from app.sources.models import DataSourceCreate, ProviderType


def test_namespacing_round_trips():
    namespaced = _namespaced_name("generic_mcp-abc123", "execute_sql")
    assert namespaced == "generic_mcp-abc123__execute_sql"
    source_id, tool_name = _split_namespaced_name(namespaced)
    assert source_id == "generic_mcp-abc123"
    assert tool_name == "execute_sql"


@pytest.fixture
def source_and_state(isolated_env, monkeypatch):
    from app.dependencies import get_source_repository

    monkeypatch.setenv("DEFAULT_USER_DISPLAY_NAME", "Mike")
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
            description="CRM and sales data",
            business_domain="sales",
        )
    )

    state = {
        "session_id": "s1",
        "messages": [HumanMessage(content="What was our sales revenue?")],
        "user_display_name": "Mike",
        "user_timezone": "UTC",
        "preferred_language": "en",
        "active_source_ids": [source.id],
        "summary": "",
        "last_response": None,
    }
    yield source, state
    reset_groq_client()


def test_no_tools_available_returns_deterministic_fallback_without_calling_groq(
    source_and_state,
):
    source, state = source_and_state

    with patch("app.agent.nodes.tool_executor.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[])
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

        result = tool_executor(state)

        assert "couldn't reach any usable tools" in result["last_response"]
        MockGroq.return_value.chat.completions.create.assert_not_called()


def test_single_tool_call_then_final_answer(source_and_state):
    source, state = source_and_state

    fake_tool = ToolInfo(
        source_id=source.id,
        tool_name="execute_sql",
        description="Run SQL",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    with patch("app.agent.nodes.tool_executor.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock(
            return_value=ToolCallResult(
                source_id=source.id,
                tool_name="execute_sql",
                content='{"revenue": 1000000}',
                is_error=False,
            )
        )
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

        # First Groq call: requests a tool call. Second: gives final answer.
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = _namespaced_name(source.id, "execute_sql")
        tool_call.function.arguments = json.dumps({"query": "SELECT revenue"})

        first_response = MagicMock()
        first_response.choices = [
            MagicMock(message=MagicMock(content=None, tool_calls=[tool_call]))
        ]
        second_response = MagicMock()
        second_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="Your sales revenue was $1,000,000.", tool_calls=None
                )
            )
        ]
        MockGroq.return_value.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        result = tool_executor(state)

        assert result["last_response"] == "Your sales revenue was $1,000,000."
        manager.call_tool.assert_called_once_with(
            source.id, "execute_sql", {"query": "SELECT revenue"}
        )
        assert len(result["messages"]) == 2  # original human msg + final AI reply


def test_tool_error_is_fed_back_and_model_can_still_answer(source_and_state):
    source, state = source_and_state
    fake_tool = ToolInfo(source_id=source.id, tool_name="execute_sql", description="")

    with patch("app.agent.nodes.tool_executor.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock(
            return_value=ToolCallResult(
                source_id=source.id,
                tool_name="execute_sql",
                content="",
                is_error=True,
                error="connection timed out",
            )
        )
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = _namespaced_name(source.id, "execute_sql")
        tool_call.function.arguments = "{}"

        first_response = MagicMock()
        first_response.choices = [
            MagicMock(message=MagicMock(content=None, tool_calls=[tool_call]))
        ]
        second_response = MagicMock()
        second_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="I couldn't reach that data source right now.",
                    tool_calls=None,
                )
            )
        ]
        MockGroq.return_value.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        result = tool_executor(state)

        assert "couldn't reach" in result["last_response"]
        second_call_kwargs = MockGroq.return_value.chat.completions.create.call_args_list[1].kwargs
        tool_messages = [
            m for m in second_call_kwargs["messages"] if m.get("role") == "tool"
        ]
        assert any("connection timed out" in m["content"] for m in tool_messages)


def test_max_iterations_produces_fallback_message(source_and_state):
    source, state = source_and_state
    fake_tool = ToolInfo(source_id=source.id, tool_name="execute_sql", description="")

    with patch("app.agent.nodes.tool_executor.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock(
            return_value=ToolCallResult(
                source_id=source.id, tool_name="execute_sql", content="{}"
            )
        )
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

        tool_call = MagicMock()
        tool_call.id = "call_x"
        tool_call.function.name = _namespaced_name(source.id, "execute_sql")
        tool_call.function.arguments = "{}"
        looping_response = MagicMock()
        looping_response.choices = [
            MagicMock(message=MagicMock(content=None, tool_calls=[tool_call]))
        ]
        MockGroq.return_value.chat.completions.create.return_value = looping_response

        result = tool_executor(state)

        assert "step limit" in result["last_response"]
        assert (
            MockGroq.return_value.chat.completions.create.call_count
            == MAX_TOOL_ITERATIONS
        )
# --- Phase 7: SQL safety validation wired into the tool loop ---------------


def test_unsafe_sql_is_blocked_before_reaching_mcp_tool(source_and_state):
    """The core Phase 7 guarantee: a DROP TABLE (or any unsafe SQL) the
    model tries to execute via a SQL-shaped tool argument must NEVER reach
    manager.call_tool — it gets intercepted by the validator, and the
    rejection reason is fed back to the model as the "tool result" so the
    model can react instead of the call silently vanishing."""
    source, state = source_and_state
    fake_tool = ToolInfo(
        source_id=source.id,
        tool_name="execute_sql",
        description="Run SQL",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    with patch("app.agent.nodes.tool_executor.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock()  # must never be called
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

        malicious_call = MagicMock()
        malicious_call.id = "call_1"
        malicious_call.function.name = _namespaced_name(source.id, "execute_sql")
        malicious_call.function.arguments = json.dumps(
            {"query": "SELECT 1; DROP TABLE sales;"}
        )

        first_response = MagicMock()
        first_response.choices = [
            MagicMock(message=MagicMock(content=None, tool_calls=[malicious_call]))
        ]
        second_response = MagicMock()
        second_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="I can't run that query — it looks like it would "
                    "modify data, which isn't allowed.",
                    tool_calls=None,
                )
            )
        ]
        MockGroq.return_value.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        result = tool_executor(state)

        manager.call_tool.assert_not_called()  # blocked before reaching MCP
        assert "can't run that query" in result["last_response"]

        second_call_kwargs = MockGroq.return_value.chat.completions.create.call_args_list[1].kwargs
        tool_messages = [
            m for m in second_call_kwargs["messages"] if m.get("role") == "tool"
        ]
        assert any(
            "blocked by safety validation" in m["content"].lower()
            for m in tool_messages
        )


def test_safe_sql_still_reaches_mcp_tool_normally(source_and_state):
    """A plain SELECT must pass straight through the validator and reach
    manager.call_tool exactly as before Phase 7."""
    source, state = source_and_state
    fake_tool = ToolInfo(
        source_id=source.id,
        tool_name="execute_sql",
        description="Run SQL",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    with patch("app.agent.nodes.tool_executor.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock(
            return_value=ToolCallResult(
                source_id=source.id,
                tool_name="execute_sql",
                content='{"revenue": 500000}',
                is_error=False,
            )
        )
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

        safe_call = MagicMock()
        safe_call.id = "call_1"
        safe_call.function.name = _namespaced_name(source.id, "execute_sql")
        safe_call.function.arguments = json.dumps(
            {"query": "SELECT revenue FROM sales WHERE region = 'US'"}
        )

        first_response = MagicMock()
        first_response.choices = [
            MagicMock(message=MagicMock(content=None, tool_calls=[safe_call]))
        ]
        second_response = MagicMock()
        second_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="US revenue was $500,000.", tool_calls=None
                )
            )
        ]
        MockGroq.return_value.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        result = tool_executor(state)

        manager.call_tool.assert_called_once_with(
            source.id,
            "execute_sql",
            {"query": "SELECT revenue FROM sales WHERE region = 'US'"},
        )
        assert result["last_response"] == "US revenue was $500,000."


def test_non_sql_tool_arguments_are_not_touched_by_validator(source_and_state):
    """A tool whose arguments don't look like SQL (no query/sql/statement
    key) should never be routed through the SQL validator at all."""
    source, state = source_and_state
    fake_tool = ToolInfo(
        source_id=source.id,
        tool_name="list_tables",
        description="List tables",
        input_schema={"type": "object", "properties": {}},
    )

    with patch("app.agent.nodes.tool_executor.get_mcp_manager") as mock_get_manager, \
         patch("app.llm.groq_client.Groq") as MockGroq:
        manager = MagicMock()
        manager.discover_tools = AsyncMock(return_value=[fake_tool])
        manager.call_tool = AsyncMock(
            return_value=ToolCallResult(
                source_id=source.id,
                tool_name="list_tables",
                content='["sales", "customers"]',
                is_error=False,
            )
        )
        manager.disconnect = AsyncMock()
        mock_get_manager.return_value = manager

        call = MagicMock()
        call.id = "call_1"
        call.function.name = _namespaced_name(source.id, "list_tables")
        call.function.arguments = "{}"

        first_response = MagicMock()
        first_response.choices = [
            MagicMock(message=MagicMock(content=None, tool_calls=[call]))
        ]
        second_response = MagicMock()
        second_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="You have two tables: sales and customers.",
                    tool_calls=None,
                )
            )
        ]
        MockGroq.return_value.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        result = tool_executor(state)

        manager.call_tool.assert_called_once_with(source.id, "list_tables", {})
        assert "sales and customers" in result["last_response"]