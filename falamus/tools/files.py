"""File I/O tools: read_file / write_file / edit_file / list_dir.

Paths are resolved relative to workdir.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from falamus.prompt import frag

from .registry import Tool, ToolResult

_MAX_READ = 100_000  # max chars per read; truncated beyond this
_MAX_DIFF = 60       # max diff lines to show
# large-file handling: rather than a hard WRITE cap (which conflicts with the read-chunk size once output
# expands, e.g. a translation), the reliable approach is CHUNKED READING — read_file requires offset/limit so
# the model processes a big file in segments (read a chunk -> transform -> append), keeping each write small.
_READ_CHUNK_DEFAULT = 8_000   # default suggested read-chunk size (chars); config: [tools] read_chunk_chars


def _make_diff(old: str, new: str, path: str) -> str:
    """Produce a unified diff (for UI +green/-red coloring); truncated if long."""
    diff = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
    ))
    if not diff:
        return ""
    if len(diff) > _MAX_DIFF:
        diff = diff[:_MAX_DIFF] + [f"… (+{len(diff) - _MAX_DIFF} more diff lines)"]
    return "\n".join(diff)


def _resolve(workdir: Path, path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    # path-doubling guard: a sub-agent's CWD already IS the workspace, so if the model re-types the
    # workspace's own tail as a relative path (e.g. ".falamus/sessions/<sid>/work[/x]"), naive joining
    # gives work/.falamus/.../work and fails. Detect when the relative path's LEADING segments duplicate
    # workdir's TRAILING segments (>=2 to avoid false hits on a single common dir name) and strip them.
    parts, base = p.parts, workdir.parts
    for k in range(min(len(parts), len(base)), 1, -1):
        if parts[:k] == base[-k:]:
            rest = parts[k:]
            return workdir.joinpath(*rest) if rest else workdir
    return workdir / p


# --- packed path header (robustness for parsers that DROP a trailing `path` arg) --------------------
# Some servers' tool-call parsers (notably llama.cpp's Gemma-4 path) silently drop the LAST argument when an
# earlier value is large or symbol-heavy — and `path` is usually emitted last, so the write fails with NO
# path. Fix: carry the path at the FRONT of the value the model emits most reliably (`content`), as a header
# `<<<FILE:relative/path>>>` followed by the body. Front placement means tail-truncation can't drop it; the
# markers avoid { } " = (the chars that trip the parser). The standard separate `path` arg still works too.
_PACK_PREFIX = "<<<FILE:"
_PACK_SUFFIX = ">>>"


def _unpack(args: dict) -> tuple[str, str]:
    """Return (path, content): the path is read ONLY from a leading `<<<FILE:path>>>` header in content.

    There is NO separate path argument — the path travels at the front of `content` (the value the model
    emits most reliably), so a large or symbol-heavy write can never drop it. The header is stripped so the
    marker never lands in the file. No header → path is "" and the caller reports an error.
    """
    content = args.get("content", "")
    path = ""
    if content.startswith(_PACK_PREFIX):
        end = content.find(_PACK_SUFFIX, len(_PACK_PREFIX))
        if end != -1:
            path = content[len(_PACK_PREFIX):end].strip()
            body = content[end + len(_PACK_SUFFIX):]
            if body.startswith("\n"):   # the example shows a newline after the header; swallow exactly one
                body = body[1:]
            content = body
    return path, content


def make_tools(workdir: str | None = None, read_chunk: int = _READ_CHUNK_DEFAULT) -> list[Tool]:
    base = Path(workdir).expanduser().resolve() if workdir else Path.cwd()

    def read_file(args: dict) -> ToolResult:
        path = args["path"]
        # REQUIRED read range (no default): forces the model to consciously choose how much to read, so it
        # uses read_file as a chunking tool for big files instead of blindly pulling everything.
        if args.get("offset") is None or args.get("limit") is None:
            return ToolResult.error(
                "read_file requires BOTH 'offset' (start char; 0 = beginning) and 'limit' (chars to read; "
                "-1 = the WHOLE file). There is no default. Whole file: read_file(path, offset=0, limit=-1). "
                "Large file in chunks: read_file(path, offset=0, limit=8000), then offset=8000, …")
        try:
            offset = max(0, int(args["offset"]))
            limit = int(args["limit"])
        except (TypeError, ValueError):
            return ToolResult.error("read_file: 'offset' and 'limit' must be integers (limit -1 = whole file).")
        p = _resolve(base, path)
        if not p.exists():
            return ToolResult.error(f"File does not exist: {p}")
        if p.is_dir():
            return ToolResult.error(f"This is a directory, not a file: {p}")
        data = p.read_text(encoding="utf-8", errors="replace")
        total = len(data)
        if total and offset >= total:   # over-shot the start → clear note instead of a confusing empty read
            return ToolResult(text=f"[file: {p} | offset {offset} is past the end (total {total} chars)]\n"
                                   "Nothing to read here — you already have the whole file; do not read further.")
        # NOTE: an over-large 'limit' is safe — slicing just stops at the end and the header/next-offset
        # below reflect what was ACTUALLY read, so the model can over-specify limit without harm.
        span = (total - offset) if limit < 0 else limit
        span = max(0, min(span, _MAX_READ))          # cap each read so a huge file can't flood the context
        chunk = data[offset:offset + span]
        end = offset + len(chunk)
        # ONLY a compact header (clearly-non-content marker, not "# ") naming the slice — NO trailing hints
        # or notes appended. A weak model was copying that scaffolding verbatim into its OUTPUT file (test4);
        # the header already states the range (chars X-Y of TOTAL), so the model can compute its next offset.
        head = f"[file: {p} | chars {offset}-{end} of {total}]"
        return ToolResult(text=f"{head}\n{chunk}")

    def write_file(args: dict) -> ToolResult:
        path, content = _unpack(args)
        if not path:
            return ToolResult.error(
                "write_file needs a path. Put it at the START of content as a header: "
                'content = "<<<FILE:your/path.txt>>>" + the text.')
        p = _resolve(base, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        old = p.read_text(encoding="utf-8", errors="replace") if existed else ""
        p.write_text(content, encoding="utf-8")
        verb = "Overwrote" if existed else "Created"
        diff = _make_diff(old, content, str(p))
        head = f"{verb}: {p} ({len(content)} chars)"
        return ToolResult(text=f"{head}\n{diff}" if diff else head)

    def append_file(args: dict) -> ToolResult:
        path, content = _unpack(args)
        if not path:
            return ToolResult.error(
                "append_file needs a path. Put it at the START of content as a header: "
                'content = "<<<FILE:your/path.txt>>>" + the part to append.')
        p = _resolve(base, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        verb = "Appended to" if existed else "Created"
        return ToolResult(text=f"{verb}: {p} (+{len(content)} chars, now {p.stat().st_size} bytes)")

    def edit_file(args: dict) -> ToolResult:
        path = args["path"]
        old = args["old"]
        new = args["new"]
        p = _resolve(base, path)
        if not p.exists():
            return ToolResult.error(f"File does not exist: {p}")
        data = p.read_text(encoding="utf-8")
        count = data.count(old)
        if count == 0:
            return ToolResult.error("Could not find the text to replace ('old' not present)")
        if count > 1:
            return ToolResult.error(f"'old' appears {count} times (not unique); provide more precise text")
        updated = data.replace(old, new, 1)
        p.write_text(updated, encoding="utf-8")
        diff = _make_diff(data, updated, str(p))
        return ToolResult(text=f"Edited: {p}\n{diff}" if diff else f"Edited: {p}")

    def list_dir(args: dict) -> ToolResult:
        path = args.get("path", ".")
        p = _resolve(base, path)
        if not p.exists():
            return ToolResult.error(f"Directory does not exist: {p}")
        if not p.is_dir():
            return ToolResult.error(f"Not a directory: {p}")
        entries = []
        for e in sorted(p.iterdir()):
            kind = "/" if e.is_dir() else ""
            size = "" if e.is_dir() else f"  {e.stat().st_size}B"
            entries.append(f"{e.name}{kind}{size}")
        listing = "\n".join(entries) or "(empty directory)"
        return ToolResult(text=f"[dir: {p}]\n{listing}")

    return [
        Tool(
            name="read_file",
            description=frag("tools", "read_file").format(read_chunk=read_chunk),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "file path (relative to workdir, or absolute)"},
                    "offset": {"type": "integer", "description": "start character position; 0 = beginning of file"},
                    "limit": {"type": "integer", "description": "how many chars to read from offset; -1 = whole file"},
                },
                "required": ["path", "offset", "limit"],
            },
            handler=read_file,
            risk="low",
        ),
        Tool(
            name="write_file",
            description=frag("tools", "write_file"),
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string",
                                "description": "the file path AND text together: <<<FILE:relative/path.txt>>> "
                                               "immediately followed by the content to write"},
                },
                "required": ["content"],
            },
            handler=write_file,
            risk="medium",
        ),
        Tool(
            name="append_file",
            description=frag("tools", "append_file"),
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string",
                                "description": "the file path AND text together: <<<FILE:relative/path.txt>>> "
                                               "immediately followed by the part to append"},
                },
                "required": ["content"],
            },
            handler=append_file,
            risk="medium",
        ),
        Tool(
            name="edit_file",
            description=frag("tools", "edit_file"),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "file path"},
                    "old": {"type": "string", "description": "the original text to replace (must be unique in the file)"},
                    "new": {"type": "string", "description": "the replacement text"},
                },
                "required": ["path", "old", "new"],
            },
            handler=edit_file,
            risk="medium",
        ),
        Tool(
            name="list_dir",
            description=frag("tools", "list_dir"),
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "directory path (defaults to workdir)"}},
                "required": [],
            },
            handler=list_dir,
            risk="low",
        ),
    ]
