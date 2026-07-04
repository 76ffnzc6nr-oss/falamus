"""Image-viewing tool: view_image.

Loads a local image file and embeds it as a data URL in the tool result, so a
(multimodal) model can "see" the image (a role:tool message carrying image_url).
"""

from __future__ import annotations

import base64
from pathlib import Path

from falamus.prompt import frag

from .registry import Tool, ToolResult

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
_MAX_BYTES = 12 * 1024 * 1024  # 12MB cap


def make_tools(workdir: str | None = None) -> list[Tool]:
    base = Path(workdir).expanduser().resolve() if workdir else Path.cwd()

    def view_image(args: dict) -> ToolResult:
        path = args["path"]
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = base / p
        if not p.exists():
            return ToolResult.error(f"Image does not exist: {p}")
        mime = _MIME.get(p.suffix.lower())
        if mime is None:
            return ToolResult.error(f"Unsupported image format: {p.suffix}")
        raw = p.read_bytes()
        if len(raw) > _MAX_BYTES:
            return ToolResult.error(f"Image too large ({len(raw)} bytes), limit {_MAX_BYTES}")
        b64 = base64.b64encode(raw).decode()
        url = f"data:{mime};base64,{b64}"
        return ToolResult(text=f"Image loaded: {p}", images=[url])

    return [
        Tool(
            name="view_image",
            description=frag("tools", "view_image"),
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "image file path"}},
                "required": ["path"],
            },
            handler=view_image,
            risk="low",
        ),
    ]
