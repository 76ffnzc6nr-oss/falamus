"""MCP **Client** transport — connect to an external MCP server over stdio so falamus can use its tools.

Layer 1 of the client direction (falamus consumes external MCP servers). Spawns the server as a subprocess
and speaks newline-delimited JSON-RPC 2.0 over its stdin/stdout — zero third-party deps. Lazy: `start()`
(spawn + handshake + tools/list) runs on first use. The tool-bridge into falamus's registry is P2; the
config CLI + safety gate are P3.

    c = McpClient(["ssh", "user@pi", "pi-io-mcp"], name="pi-io")
    c.start()                       # spawn + handshake + discover tools
    for t in c.tools: ...           # {name, description, inputSchema}
    text, is_error = c.call_tool("gpio_write", {"pin": 17, "value": 1})
    c.close()
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

_PROTOCOL = "2024-11-05"


class McpError(Exception):
    """An external MCP server returned an error, closed, or misbehaved."""


class McpClient:
    def __init__(self, command: list[str], name: str = "") -> None:
        self.command = command
        self.name = name or (command[0] if command else "mcp")
        self.server_info: dict[str, Any] = {}
        self.tools: list[dict[str, Any]] = []
        self._p: subprocess.Popen | None = None
        self._id = 0

    # ---- lifecycle ------------------------------------------------------
    def start(self) -> None:
        """Spawn the server, complete the MCP handshake, and discover its tools. Idempotent (lazy)."""
        if self._p is not None:
            return
        try:
            self._p = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError as e:
            raise McpError(f"could not start MCP server {self.name!r}: {e}") from e
        init = self._request("initialize", {"protocolVersion": _PROTOCOL, "capabilities": {},
                                            "clientInfo": {"name": "falamus", "version": _version()}})
        self.server_info = init.get("serverInfo", {})
        self._notify("notifications/initialized")
        self.tools = self._request("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Call an external tool; return (text, is_error). Auto-starts if needed."""
        self.start()
        res = self._request("tools/call", {"name": name, "arguments": arguments})
        content = res.get("content") or []
        text = content[0].get("text", "") if content else ""
        return text, bool(res.get("isError"))

    def close(self) -> None:
        if self._p is None:
            return
        try:
            if self._p.stdin:
                self._p.stdin.close()
            self._p.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            self._p.kill()
        self._p = None

    # ---- JSON-RPC over the subprocess -----------------------------------
    def _request(self, method: str, params: dict) -> dict:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        while True:                                   # skip any interleaved notifications; match our id
            msg = self._recv()
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise McpError(f"{self.name}/{method}: {msg['error'].get('message', 'error')}")
                return msg.get("result", {}) or {}

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def _send(self, msg: dict) -> None:
        if not (self._p and self._p.stdin):
            raise McpError(f"{self.name}: not connected")
        self._p.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._p.stdin.flush()

    def _recv(self) -> dict:
        if not (self._p and self._p.stdout):
            raise McpError(f"{self.name}: not connected")
        line = self._p.stdout.readline()
        if not line:
            raise McpError(f"{self.name}: server closed the connection")
        return json.loads(line)


def _version() -> str:
    from falamus.version import __version__
    return __version__
