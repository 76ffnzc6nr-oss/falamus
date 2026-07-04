"""The external MCP server list — `~/.config/falamus/mcp_servers.json` (standard `mcpServers` shape).

Configuring a server here is how a user opts IN to the MCP client: any server listed is connected (lazily,
when a session starts) and its tools are bridged into the agent's toolbox. Empty file = no external tools.
"""
from __future__ import annotations

import json
from pathlib import Path


def mcp_servers_path() -> Path:
    from falamus.settings import CONFIG_PATH
    return CONFIG_PATH.parent / "mcp_servers.json"


def load_mcp_servers() -> dict[str, dict]:
    """{name: {"command": str, "args": [str, …]}}. Missing/broken file → empty (never raises)."""
    p = mcp_servers_path()
    if not p.is_file():
        return {}
    try:
        servers = json.loads(p.read_text(encoding="utf-8")).get("mcpServers", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    return servers if isinstance(servers, dict) else {}


def save_mcp_servers(servers: dict[str, dict]) -> None:
    p = mcp_servers_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"mcpServers": servers}, ensure_ascii=False, indent=2), encoding="utf-8")


def add_mcp_server(name: str, command: str, args: list[str]) -> None:
    servers = load_mcp_servers()
    servers[name] = {"command": command, "args": list(args)}
    save_mcp_servers(servers)


def remove_mcp_server(name: str) -> bool:
    servers = load_mcp_servers()
    if name not in servers:
        return False
    del servers[name]
    save_mcp_servers(servers)
    return True


def server_command(spec: dict) -> list[str]:
    """The full command list to spawn a server from its {command, args} spec."""
    return [spec["command"], *spec.get("args", [])]
