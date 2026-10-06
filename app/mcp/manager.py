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
        try:
            pat = self._sources.resolve_credential(source_id)
        except CredentialNotFoundError as exc:
            # The most common cause by far: CREDENTIAL_STORE_BACKEND=env,
            # where a PAT only lives in the memory of the process that
            # registered it — a separate command (even moments later)
            # never sees it. Second most common: LOCAL_CREDENTIAL_STORE_KEY
            # was changed after the source was created, so the old
            # encrypted value can no longer be decrypted. Spell both out
            # rather than surfacing the bare KeyError, which is just the
            # credential_ref and explains nothing on its own.
            raise MCPConnectionError(
                f"No stored credential for source '{source_id}'. If "
                "CREDENTIAL_STORE_BACKEND=env, a PAT only persists for the "
                "lifetime of the process that registered it — switch to "
                "CREDENTIAL_STORE_BACKEND=local_file (with a "
                "LOCAL_CREDENTIAL_STORE_KEY set) for it to survive across "
                "commands/restarts. If you're already using local_file, "
                "check LOCAL_CREDENTIAL_STORE_KEY hasn't changed since "
                "this source was created — re-add the source if it has."
            ) from exc

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