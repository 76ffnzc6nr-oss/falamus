"""Tool framework.

Design:
  - `Tool`        : a tool definition (name, description, JSON schema, handler).
  - `ToolResult`  : a tool's result, can carry text and images (for view_image / screenshots).
  - `ToolRegistry`: registers tools, builds the tools schema for the model, executes a ToolCall.

When a tool result is returned to the model it uses the verified format:
  - plain text  → `content` is a string.
  - with images → `content` is an array `[{"type":"text",...},{"type":"image_url",...}]`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ToolCall comes from the client layer; to avoid a hard dependency we use duck typing
# in annotations. We only need three attributes: .id / .name / .arguments


@dataclass
class ToolResult:
    """A tool execution result."""

    text: str = ""
    images: list[str] = field(default_factory=list)  # data URL (data:image/...;base64,xxx)
    is_error: bool = False

    def to_message(self, call_id: str, name: str) -> dict[str, Any]:
        """Convert into a llama.cpp/OpenAI-compatible role:tool message."""
        if self.images:
            content: Any = []
            if self.text:
                content.append({"type": "text", "text": self.text})
            for url in self.images:
                content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            content = self.text
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }

    @classmethod
    def error(cls, msg: str) -> ToolResult:
        return cls(text=f"[error] {msg}", is_error=True)


@dataclass
class Tool:
    """A single tool definition."""

    name: str
    description: str           # the "intro" shown to the model (used as the full description for now)
    parameters: dict[str, Any]  # JSON schema
    handler: Callable[[dict[str, Any]], ToolResult]
    risk: str = "low"          # low | medium | high (used by the safety policy)

    def schema(self) -> dict[str, Any]:
        """Produce the OpenAI tools format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Tool registry."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # safety guard: guard(tool, args) -> ToolResult|None; None = allow, ToolResult = intercept
        self.guard: Callable[[Tool, dict[str, Any]], ToolResult | None] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def to_openai_tools(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """Build the tools list sent to the model; `allowed` can restrict it (for sub-agents)."""
        return [
            t.schema()
            for name, t in self._tools.items()
            if allowed is None or name in allowed
        ]

    def execute(self, call: Any) -> ToolResult:
        """Run the tool matching the model's ToolCall, catching exceptions uniformly."""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {call.name}")
        args = call.arguments if isinstance(call.arguments, dict) else {}
        # detect truncated / bad-JSON arguments (the client puts them in _raw); give clear guidance to avoid a loop
        if "_raw" in args:
            return ToolResult.error(
                f"The arguments for {call.name} are not valid JSON (likely truncated by overly long output). "
                "Write large output in PARTS: write_file the first part, then append_file each next part."
            )
        # validate REQUIRED arguments → return an actionable message instead of a raw KeyError, so the model
        # can fix the call next turn (rather than re-issuing the same broken one). Generic: uses each tool's
        # declared JSON-schema `required`, so it applies to every current and future tool.
        required = tool.parameters.get("required", []) if isinstance(tool.parameters, dict) else []
        missing = [k for k in required if k not in args or args.get(k) is None]
        if missing:
            sig = ", ".join(f"{k}=..." for k in required)
            return ToolResult.error(
                f"{call.name} is missing required argument(s): {', '.join(missing)}. "
                f"Retry with all required arguments — {call.name}({sig})."
            )
        # safety guard: block / require confirmation
        if self.guard is not None:
            blocked = self.guard(tool, args)
            if blocked is not None:
                return blocked
        try:
            return tool.handler(args)
        except Exception as e:  # noqa: BLE001 — tool errors must go back to the model, not break the loop
            return ToolResult.error(f"{call.name} failed: {type(e).__name__}: {e}")


def default_registry(workdir: str | None = None) -> ToolRegistry:
    """Build a registry containing all built-in tools.

    workdir is the base directory for the file/CLI tools (defaults to the current working directory).
    """
    from . import cli, files, image

    reg = ToolRegistry()
    reg.register_all(files.make_tools(workdir))
    reg.register_all(cli.make_tools(workdir))
    reg.register_all(image.make_tools(workdir))
    return reg


if __name__ == "__main__":
    import json

    reg = default_registry()
    print("Registered tools:", reg.names())
    print("\ntools schema sent to the model (excerpt):")
    print(json.dumps(reg.to_openai_tools()[:2], ensure_ascii=False, indent=2))
