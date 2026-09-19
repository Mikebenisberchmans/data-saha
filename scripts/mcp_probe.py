"""
Dev-only script to connect to a configured MCP source, discover its tools,
and optionally call one — without needing the full agent/graph.

Usage:
    python -m scripts.mcp_probe --id generic_mcp-63d1b251
    python -m scripts.mcp_probe --id generic_mcp-63d1b251 \
        --call execute_sql --args '{"query": "select 1"}'
"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.dependencies import get_mcp_manager


async def _run(args: argparse.Namespace) -> None:
    manager = get_mcp_manager()

    print(f"Connecting to source '{args.id}' ...")
    health = await manager.health_check(args.id)
    status = "OK" if health.healthy else "FAILED"
    suffix = f" ({health.message})" if health.message else ""
    print(f"Health: {status}{suffix}")
    if not health.healthy:
        return

    tools = await manager.discover_tools(args.id)
    print(f"\nDiscovered {len(tools)} tool(s):")
    for t in tools:
        print(f"  - {t.tool_name}: {t.description}")
        print(f"    schema: {json.dumps(t.input_schema)}")

    if args.call:
        call_args = json.loads(args.args) if args.args else {}
        print(f"\nCalling '{args.call}' with {call_args} ...")
        result = await manager.call_tool(args.id, args.call, call_args)
        print("is_error:", result.is_error)
        if result.error:
            print("error:", result.error)
        print("content:")
        print(result.content)

    await manager.disconnect(args.id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a configured MCP source.")
    parser.add_argument(
        "--id",
        required=True,
        help="Data source id (see: python -m scripts.manage_sources list)",
    )
    parser.add_argument("--call", default=None, help="Tool name to call after discovery")
    parser.add_argument("--args", default=None, help="JSON arguments for --call")
    parsed = parser.parse_args()
    asyncio.run(_run(parsed))


if __name__ == "__main__":
    main()