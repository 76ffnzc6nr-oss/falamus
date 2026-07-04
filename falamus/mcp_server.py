"""falamus MCP **Server** (Layer 3) — expose falamus as a tool other agents can call.

A minimal, hand-rolled MCP server over stdio: newline-delimited JSON-RPC 2.0, zero third-party deps. It
wraps the Session API (P1) + policy (P2). Transport = stdio: the CLIENT spawns this process (`falamus --mcp`,
or `ssh pi falamus --mcp` for a remote box) and talks over stdin/stdout — no port, no daemon.

Tools (concise results — the transcript is NOT streamed back; deliverable FILES land on disk):
  falamus_run(task, overrides?, auto_approve?)  run one multi-agent turn → {reply, deliverables, status, sid}
  falamus_confirm(id, decision)                 answer a pending confirmation (interactive mode)
  falamus_status()                              current session id / model
  falamus_resume(sid)                           reload a previous session; later runs continue it

Resources (read on demand — kept OUT of the reply to save the caller's tokens):
  falamus://session/transcript   the last run's structured events
  falamus://session/errors       the error log
  falamus://deliverable/<path>   a deliverable file's content (restricted to the working directory)

Confirmations: with `auto_approve=true` (default) dangerous actions are auto-approved (bounded by the strict
server-fixed config + the blacklist) and audited. With `auto_approve=false` the run PAUSES and the caller
answers each one via falamus_confirm — the caller IS the human.
"""
from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

_PROTOCOL = "2024-11-05"     # MCP protocol version this server implements

_TOOLS = [
    {
        "name": "falamus_run",
        "description": ("Run ONE task through the local falamus multi-agent (main + sub-agents) and return "
                        "the final reply, the deliverable file paths it produced, and a status. Files are "
                        "written to disk in the server's working directory. With auto_approve=false the run "
                        "pauses on each dangerous action and returns a pending_confirmation for you to answer "
                        "with falamus_confirm."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "the task, in natural language"},
                "overrides": {
                    "type": "object",
                    "description": ("optional per-session behaviour/quality/cost settings. Safe keys: "
                                    "max_depth, max_iters_main, max_iters_sub, thinking (off/low/medium/high), "
                                    "auto_compact, compact_threshold, read_chunk_chars, repeat_penalty, "
                                    "repeat_last_n. Capability/security keys (persistent_interactive_shell, allowed_paths, "
                                    "workdir, model, …) are server-fixed and ignored (reported in "
                                    "'rejected_overrides'). LOCKED at session creation: one connection = one "
                                    "session, so overrides take effect on the FIRST run; different overrides on "
                                    "a later run are ignored (reported in 'overrides_ignored') — open a new "
                                    "connection to change them."),
                },
                "auto_approve": {
                    "type": "boolean",
                    "description": "true (default) auto-approves dangerous actions; false asks you per action.",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "falamus_confirm",
        "description": ("Answer a pending confirmation from a falamus_run started with auto_approve=false. "
                        "Returns the next pending_confirmation, or the final run result once the run finishes."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "the pending confirmation's id"},
                "decision": {"type": "boolean", "description": "true to allow, false to reject"},
            },
            "required": ["id", "decision"],
        },
    },
    {
        "name": "falamus_status",
        "description": "Return the current session id and the model falamus is running.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "falamus_resume",
        "description": "Reload a previous session by id (its conversation tail is replayed); later runs continue it.",
        "inputSchema": {
            "type": "object",
            "properties": {"sid": {"type": "string", "description": "the session id to resume"}},
            "required": ["sid"],
        },
    },
]


def _ok(mid: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def _content(obj: dict, is_error: bool = False) -> dict:
    """An MCP tool result: a single text block carrying JSON (widely compatible)."""
    return {"content": [{"type": "text", "text": json.dumps(obj, ensure_ascii=False)}], "isError": is_error}


class _RunBridge:
    """Runs Session.run on a background thread so the caller can answer confirmations BETWEEN MCP calls. The
    session's confirm callback is `confirm` (runs on the worker thread): it parks the request and blocks until
    `answer` delivers a decision. The main thread drives it with start() / wait() / answer()."""

    def __init__(self) -> None:
        self._pending: dict | None = None
        self._answer = False
        self._outcome: tuple[str, Any] | None = None    # ("done", RunResult) | ("error", str)
        self._to_main = threading.Event()
        self._to_worker = threading.Event()
        self._cid = 0
        self._lock = threading.Lock()

    def confirm(self, tool: Any, args: dict, reason: str) -> bool:     # runs on the worker thread
        with self._lock:
            self._cid += 1
            self._pending = {"id": str(self._cid), "tool": getattr(tool, "name", str(tool)), "reason": reason}
        self._to_worker.clear()
        self._to_main.set()          # wake the main thread: a confirm is pending
        self._to_worker.wait()       # block until answered
        return self._answer

    def start(self, session: Any, task: str) -> None:
        def _work() -> None:
            try:
                self._outcome = ("done", session.run(task))
            except Exception as e:  # noqa: BLE001
                self._outcome = ("error", str(e))
            self._to_main.set()      # wake the main thread: the run finished
        threading.Thread(target=_work, daemon=True).start()

    def wait(self) -> tuple[str, Any]:   # main thread: block until pending-confirm OR run-done
        self._to_main.wait()
        self._to_main.clear()
        if self._outcome is not None:
            return self._outcome
        return ("pending", self._pending)

    def answer(self, cid: str, decision: bool) -> bool:
        with self._lock:
            if not self._pending or self._pending["id"] != str(cid):
                return False         # stale / unknown id
            self._pending = None
        self._answer = bool(decision)
        self._to_main.clear()
        self._to_worker.set()        # wake the worker with the decision
        return True


class MCPServer:
    """Dispatches MCP JSON-RPC messages against a falamus Session, built LAZILY (so the handshake works even
    if the model server is momentarily unreachable). Holds a per-run confirm bridge + the last result."""

    def __init__(self, session: Any = None,
                 session_factory: Callable[[dict, Callable], Any] | None = None) -> None:
        self._session = session
        self._factory = session_factory            # (overrides: dict, on_confirm) -> Session
        self._session_overrides: dict | None = None
        self._bridge: _RunBridge | None = None     # active during an interactive (auto_approve=false) run
        self._last_result: Any = None              # last RunResult (for the resources)
        self._workdir: str | None = None
        self._override_note: str | None = None     # set when a later run's overrides were ignored (locked)

    def _dispatch_confirm(self, tool: Any, args: dict, reason: str) -> bool:
        b = self._bridge
        return b.confirm(tool, args, reason) if b is not None else True   # no bridge → auto-approve

    def _sess(self, overrides: dict | None = None) -> Any:
        # ONE connection = ONE session: overrides are LOCKED when the session is first created. A later run
        # passing DIFFERENT overrides doesn't silently rebuild (that would lose the conversation) — they are
        # ignored and reported. To use different settings, open a new connection.
        self._override_note = None
        if self._factory is None:
            if self._session is None:
                raise RuntimeError("no session")
            return self._session
        if self._session is None:
            self._session = self._factory(overrides or {}, self._dispatch_confirm)
            self._session_overrides = overrides or {}
        elif overrides and overrides != self._session_overrides:
            self._override_note = ("overrides ignored — this session was created with "
                                   f"{self._session_overrides or 'defaults'}; open a new connection to change them")
        self._workdir = getattr(getattr(self._session, "backend", None), "workdir", None)
        return self._session

    # ---- handshake / dispatch -------------------------------------------
    def handle(self, msg: dict) -> dict | None:
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            return _ok(mid, {"protocolVersion": _PROTOCOL,
                             "capabilities": {"tools": {}, "resources": {}},
                             "serverInfo": {"name": "falamus", "version": _version()}})
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return _ok(mid, {})
        if method == "tools/list":
            return _ok(mid, {"tools": _TOOLS})
        if method == "tools/call":
            return self._call(mid, msg.get("params") or {})
        if method == "resources/list":
            return _ok(mid, {"resources": self._resources_list()})
        if method == "resources/read":
            return self._read(mid, msg.get("params") or {})
        return _err(mid, -32601, f"method not found: {method}")

    # ---- tools ----------------------------------------------------------
    def _call(self, mid: Any, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if name == "falamus_run":
                s = self._sess(args.get("overrides"))
                if bool(args.get("auto_approve", True)):
                    self._bridge = None
                    return self._finish(mid, s, s.run(str(args["task"])))
                self._bridge = _RunBridge()
                self._bridge.start(s, str(args["task"]))
                return self._after_wait(mid, s)
            if name == "falamus_confirm":
                if self._bridge is None:
                    return _ok(mid, _content({"error": "no confirmation pending"}, is_error=True))
                if not self._bridge.answer(str(args["id"]), bool(args.get("decision", False))):
                    return _ok(mid, _content({"error": "unknown or stale confirmation id"}, is_error=True))
                return self._after_wait(mid, self._session)
            if name == "falamus_status":
                s = self._sess()
                return _ok(mid, _content({"sid": s.sid, "model": _model_of(s)}))
            if name == "falamus_resume":
                s = self._sess()
                s.resume(str(args["sid"]))
                return _ok(mid, _content({"sid": s.sid, "resumed": True}))
            return _ok(mid, _content({"error": f"unknown tool: {name}"}, is_error=True))
        except Exception as e:  # noqa: BLE001 — surface any run/connect error, don't crash the server
            return _ok(mid, _content({"error": str(e)}, is_error=True))

    def _finish(self, mid: Any, s: Any, res: Any) -> dict:
        self._last_result = res
        body = {"reply": res.reply, "deliverables": res.deliverables, "status": res.status, "sid": res.sid}
        rejected = getattr(s, "rejected_overrides", None)
        if rejected:
            body["rejected_overrides"] = rejected
        if self._override_note:
            body["overrides_ignored"] = self._override_note
        return _ok(mid, _content(body))

    def _after_wait(self, mid: Any, s: Any) -> dict:
        state, payload = self._bridge.wait()  # type: ignore[union-attr]
        if state == "pending":
            return _ok(mid, _content({"status": "pending_confirmation", "pending_confirmation": payload}))
        self._bridge = None
        if state == "error":
            return _ok(mid, _content({"error": payload}, is_error=True))
        return self._finish(mid, s, payload)

    # ---- resources ------------------------------------------------------
    def _resources_list(self) -> list[dict]:
        res = [{"uri": "falamus://session/errors", "name": "error log", "mimeType": "text/markdown"}]
        if self._last_result is not None:
            res.append({"uri": "falamus://session/transcript", "name": "last run transcript",
                        "mimeType": "application/json"})
            for d in self._last_result.deliverables:
                res.append({"uri": f"falamus://deliverable/{d}", "name": d, "mimeType": "text/plain"})
        return res

    def _read(self, mid: Any, params: dict) -> dict:
        uri = str(params.get("uri", ""))
        try:
            text, mime = self._read_resource(uri)
        except (ValueError, OSError) as e:
            return _err(mid, -32602, f"cannot read resource {uri}: {e}")
        return _ok(mid, {"contents": [{"uri": uri, "mimeType": mime, "text": text}]})

    def _read_resource(self, uri: str) -> tuple[str, str]:
        if uri == "falamus://session/transcript":
            events = self._last_result.events if self._last_result is not None else []
            return json.dumps(events, ensure_ascii=False, default=str), "application/json"
        if uri == "falamus://session/errors":
            p = Path(self._workdir) / ".falamus" / "error_log.md" if self._workdir else None
            return (p.read_text(encoding="utf-8") if p and p.is_file() else ""), "text/markdown"
        prefix = "falamus://deliverable/"
        if uri.startswith(prefix):
            return self._read_deliverable(uri[len(prefix):]), "text/plain"
        raise ValueError("unknown resource uri")

    def _read_deliverable(self, rel: str) -> str:
        if not self._workdir:
            raise ValueError("no working directory")
        base = Path(self._workdir).resolve()
        target = (base / rel).resolve()
        if target != base and base not in target.parents:     # escape guard: stay inside the workdir
            raise ValueError("path is outside the working directory")
        return target.read_text(encoding="utf-8")

    # ---- transport ------------------------------------------------------
    def serve(self, stdin: TextIO, stdout: TextIO) -> None:
        """Read newline-delimited JSON-RPC from stdin, write responses to stdout."""
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle(msg)
            if resp is not None:
                stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                stdout.flush()


def _version() -> str:
    from falamus.version import __version__
    return __version__


def _model_of(session: Any) -> str:
    try:
        return session.backend.client.model or ""
    except AttributeError:
        return ""


def main() -> None:
    """`falamus --mcp`: serve one falamus session over stdio to whichever client spawned this process."""
    from falamus.session_api import Session
    from falamus.settings import Config

    def factory(overrides: dict, on_confirm: Callable) -> Session:
        cfg = Config.load()
        audit = str(Path(cfg.workdir) / ".falamus" / "mcp_audit.jsonl") if cfg.workdir else None
        return Session.connect(cfg, overrides=overrides, on_confirm=on_confirm, audit_path=audit)

    MCPServer(session_factory=factory).serve(sys.stdin, sys.stdout)
