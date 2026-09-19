from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp.connection import MCPConnection, MCPConnectionError
from app.sources.models import DataSourceConfig, ProviderType


def _make_source(source_id: str = "generic_mcp-test1") -> DataSourceConfig:
    return DataSourceConfig(
        id=source_id,
        credential_ref=source_id,
        display_name="Test Source",
        provider=ProviderType.GENERIC_MCP,
        mcp_url="https://example.com/mcp",
        description="test",
        business_domain=None,
        enabled=True,
    )


def test_not_connected_raises_on_discover():
    conn = MCPConnection(source=_make_source(), pat="fake-pat")
    with pytest.raises(MCPConnectionError):
        import asyncio

        asyncio.run(conn.discover_tools())


def test_repr_never_leaks_pat():
    conn = MCPConnection(source=_make_source(), pat="super-secret-pat-value")
    assert "super-secret-pat-value" not in repr(conn)
    assert "super-secret-pat-value" not in str(conn)


@pytest.mark.asyncio
async def test_connect_uses_pat_as_bearer_header_and_never_logs_it(caplog):
    source = _make_source()
    conn = MCPConnection(source=source, pat="super-secret-pat-value")

    fake_session = AsyncMock()
    fake_session.initialize = AsyncMock()

    with patch("app.mcp.connection.create_mcp_http_client") as mock_create_client, \
         patch("app.mcp.connection.streamable_http_client") as mock_transport, \
         patch("app.mcp.connection.ClientSession") as MockClientSession:

        mock_http_client = AsyncMock()
        mock_create_client.return_value = mock_http_client

        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.return_value = (MagicMock(), MagicMock())
        mock_transport.return_value = mock_transport_cm

        MockClientSession.return_value.__aenter__.return_value = fake_session

        await conn.connect()

        _, kwargs = mock_create_client.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer super-secret-pat-value"

    assert conn.is_connected
    for record in caplog.records:
        assert "super-secret-pat-value" not in record.getMessage()


@pytest.mark.asyncio
async def test_discover_tools_maps_sdk_tools_to_tool_info():
    source = _make_source()
    conn = MCPConnection(source=source, pat="fake-pat")
    conn._connected = True

    fake_tool = MagicMock(name="execute_sql", description="Run SQL")
    fake_tool.name = "execute_sql"
    fake_tool.description = "Run SQL"
    fake_tool.input_schema = {"type": "object"}

    fake_session = AsyncMock()
    fake_list_result = MagicMock()
    fake_list_result.tools = [fake_tool]
    fake_session.list_tools = AsyncMock(return_value=fake_list_result)
    conn._session = fake_session

    tools = await conn.discover_tools()
    assert len(tools) == 1
    assert tools[0].tool_name == "execute_sql"
    assert tools[0].source_id == source.id
    assert tools[0].description == "Run SQL"


@pytest.mark.asyncio
async def test_call_tool_returns_error_result_on_exception():
    source = _make_source()
    conn = MCPConnection(source=source, pat="fake-pat")
    conn._connected = True

    fake_session = AsyncMock()
    fake_session.call_tool = AsyncMock(side_effect=RuntimeError("boom"))
    conn._session = fake_session

    result = await conn.call_tool("execute_sql", {"query": "select 1"})
    assert result.is_error is True
    assert "boom" in result.error
    assert result.source_id == source.id