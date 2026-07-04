"""Bridge external MCP tools (discovered via an McpClient) into falamus `Tool`s.

Layer 2 of the client direction. Each external tool becomes a falamus Tool whose description + parameter
schema are the SERVER's own (faithful pass-through — the model sees it exactly as the server describes it,
NOT from falamus's prompt sets), and whose handler forwards a `tools/call` to that server. The tool name is
namespaced `<server>__<tool>` to avoid collisions with built-ins / other servers. Marked `risk="high"` so
the safety policy treats an external-tool call as a dangerous action (P3 gates it with a confirm).
"""
from __future__ import annotations

import re

from falamus.mcp_client import McpClient, McpError
from falamus.tools.registry import Tool, ToolResult


def _prefix(name: str) -> str:
    return re.sub(r"\W", "_", name) or "mcp"          # sanitise the server name for a valid tool prefix


def bridged_tools(client: McpClient) -> list[Tool]:
    """Return one falamus Tool per external tool the client exposes (starts/discovers the client if needed)."""
    client.start()
    return [_bridge_one(client, spec) for spec in client.tools]


def _bridge_one(client: McpClient, spec: dict) -> Tool:
    orig = spec["name"]
    schema = spec.get("inputSchema") or {"type": "object", "properties": {}}

    def handler(args: dict, _orig: str = orig) -> ToolResult:
        try:
            text, is_error = client.call_tool(_orig, args)
        except McpError as e:
            return ToolResult.error(f"[mcp:{client.name}] {e}")
        return ToolResult(text=text, is_error=is_error)

    return Tool(
        name=f"{_prefix(client.name)}__{orig}",
        description=spec.get("description") or f"external MCP tool '{orig}' (from {client.name})",
        parameters=schema,
        handler=handler,
        risk="high",           # external tool = new attack surface → dangerous-action tier (gated in P3)
    )
