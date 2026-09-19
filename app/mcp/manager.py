"""
MCPManager: lazy, source-id-keyed pool of MCPConnection objects.

Per product spec section 6, connections are NOT opened for every
configured source on every user message — only the source(s) actually
relevant to a given question get connected, and only on first use.
Connections are then reused across subsequent calls within the process,
not reopened per call.

This does not itself decide WHICH source(s) are relevant to a question —
that's the source-selection node (Phase 5). MCPManager only manages the
connect/discover/call/disconnect lifecycle once a caller has already
decided which source_id(s) it needs.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.mcp.connection import MCPConnection
from app.mcp.models import ConnectionHealth, ToolCallResult, ToolInfo
from app.sources.repository import SourceRepository

logger = get_logger(__name__)


class MCPManager:
    def __init__(self, source_repository: SourceRepository):
        self._sources = source_repository
        self._connections: dict[str, MCPConnection] = {}

    async def get_connection(self, source_id: str) -> MCPConnection:
        """Returns a connected MCPConnection for source_id, creating and
        connecting it lazily on first use. Reuses an existing connection
        if one is already live."""
        existing = self._connections.get(source_id)
        if existing is not None and existing.is_connected:
            return existing

        source = self._sources.get_source(source_id)
        pat = self._sources.resolve_credential(source_id)
        connection = MCPConnection(source=source, pat=pat)
        await connection.connect()
        self._connections[source_id] = connection
        return connection

    async def discover_tools(self, source_id: str) -> list[ToolInfo]:
        connection = await self.get_connection(source_id)
        return await connection.discover_tools()

    async def call_tool(
        self, source_id: str, tool_name: str, arguments: dict | None = None
    ) -> ToolCallResult:
        connection = await self.get_connection(source_id)
        return await connection.call_tool(tool_name, arguments)

    async def health_check(self, source_id: str) -> ConnectionHealth:
        try:
            connection = await self.get_connection(source_id)
            return await connection.health_check()
        except Exception as exc:
            return ConnectionHealth(
                source_id=source_id, healthy=False, message=str(exc)
            )

    async def disconnect(self, source_id: str) -> None:
        connection = self._connections.pop(source_id, None)
        if connection is not None:
            await connection.disconnect()

    async def disconnect_all(self) -> None:
        for source_id in list(self._connections.keys()):
            await self.disconnect(source_id)