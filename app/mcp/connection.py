"""
MCP connection lifecycle for a single configured data source.

Wraps the official Python MCP SDK's client-side ClientSession over
Streamable HTTP transport (the transport a hosted MCP server like
Supabase's exposes over an https:// URL). One MCPConnection = one live
session to one configured DataSourceConfig. Created lazily by MCPManager
(see app/mcp/manager.py) — never eagerly for every configured source, per
product spec section 6.

Security: the resolved PAT is fetched once, right before connecting, and
held only as an in-memory HTTP Authorization header for the lifetime of
this connection object. It is never logged, never included in
__repr__/__str__, and never appears in any ToolInfo/ToolCallResult
returned to callers — those only ever carry source_id (a public,
non-secret identifier).
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from app.core.logging import get_logger
from app.mcp.models import ConnectionHealth, ToolCallResult, ToolInfo
from app.sources.models import DataSourceConfig

logger = get_logger(__name__)


class MCPConnectionError(RuntimeError):
    pass


class MCPConnection:
    def __init__(self, source: DataSourceConfig, pat: str):
        self._source = source
        self._pat = pat  # in-memory only; never logged, never __repr__'d
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._connected = False

    @property
    def source_id(self) -> str:
        return self._source.id

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connected:
            return

        self._stack = AsyncExitStack()
        try:
            http_client = create_mcp_http_client(
                headers={"Authorization": f"Bearer {self._pat}"}
            )
            await self._stack.enter_async_context(http_client)

            read, write = await self._stack.enter_async_context(
                streamable_http_client(self._source.mcp_url, http_client=http_client)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            self._connected = True
            logger.info(
                "Connected to MCP source id=%s provider=%s",
                self._source.id,
                self._source.provider.value,
            )
        except Exception as exc:
            await self._safe_close()
            logger.error(
                "Failed to connect to MCP source id=%s: %s",
                self._source.id,
                type(exc).__name__,
            )
            raise MCPConnectionError(
                f"Failed to connect to source '{self._source.id}': {exc}"
            ) from exc

    async def discover_tools(self) -> list[ToolInfo]:
        """Per product spec section 7: never assume a fixed tool set.
        Whatever the server reports via list_tools() is the source of
        truth."""
        self._ensure_connected()
        result = await self._session.list_tools()
        return [
            ToolInfo(
                source_id=self._source.id,
                tool_name=t.name,
                description=t.description or "",
                input_schema=t.input_schema or {},
            )
            for t in result.tools
        ]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ToolCallResult:
        self._ensure_connected()
        try:
            result = await self._session.call_tool(tool_name, arguments or {})
            text_parts = [
                block.text
                for block in (result.content or [])
                if getattr(block, "type", None) == "text"
            ]
            return ToolCallResult(
                source_id=self._source.id,
                tool_name=tool_name,
                content="\n".join(text_parts),
                is_error=bool(getattr(result, "is_error", False)),
            )
        except Exception as exc:
            logger.error(
                "Tool call failed source_id=%s tool=%s: %s",
                self._source.id,
                tool_name,
                type(exc).__name__,
            )
            return ToolCallResult(
                source_id=self._source.id,
                tool_name=tool_name,
                content="",
                is_error=True,
                error=str(exc),
            )

    async def health_check(self) -> ConnectionHealth:
        try:
            if not self._connected:
                await self.connect()
            await self._session.list_tools()
            return ConnectionHealth(source_id=self._source.id, healthy=True)
        except Exception as exc:
            return ConnectionHealth(
                source_id=self._source.id, healthy=False, message=str(exc)
            )

    async def disconnect(self) -> None:
        await self._safe_close()
        logger.info("Disconnected from MCP source id=%s", self._source.id)

    async def _safe_close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                logger.warning(
                    "Error while closing MCP connection id=%s", self._source.id
                )
        self._stack = None
        self._session = None
        self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected or self._session is None:
            raise MCPConnectionError(
                f"Source '{self._source.id}' is not connected. Call connect() first."
            )

    def __repr__(self) -> str:  # never leak the PAT
        return f"<MCPConnection source_id={self._source.id} connected={self._connected}>"