"""Programmatic entry to a falamus session (Layer 1 of the agent interface).

A thin, transport-agnostic wrapper over `Backend.run_message`: submit a task, get a STRUCTURED result
(final reply + deliverable files + status + events), multi-turn, session-persisted. This is the seam every
outward interface wraps (the planned MCP server), and it also drives in-process prompt testing.

    from falamus.session_api import Session
    s = Session.connect(cfg)              # builds a real Backend (connects to the model server)
    res = s.run("translate docs/ to English")
    print(res.reply, res.deliverables, res.status)

No new dependencies; no change to the agent loop, prompts, or the TUI. `Session` takes any built backend,
so tests inject a fake one instead of connecting to a server.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_SESSION_INTERNALS = ".falamus"   # session store / staging workspace live here → NOT deliverables


@dataclass
class RunResult:
    """The structured outcome of one `Session.run(task)` turn."""
    reply: str                              # final reply to the user (the process is not included)
    deliverables: list[str]                 # project files created/changed this turn (relative paths)
    status: str                             # ok | degenerate | iter_limit | interrupted | error
    sid: str                                # session id (for resume)
    events: list[dict[str, Any]] = field(default_factory=list)  # tool calls / sub-agent / degen … events


def classify_status(reply: str) -> str:
    """Map a run_message reply to a status. The agent returns special bracketed prefixes for non-normal
    endings; anything else is a normal completion."""
    r = reply.lstrip()
    if r.startswith("[generation aborted: degenerate]"):
        return "degenerate"
    if r.startswith("[reached"):
        return "iter_limit"
    if r.startswith("[interrupted]"):
        return "interrupted"
    if r.startswith("[error]") or r.startswith("[server"):
        return "error"
    return "ok"


def snapshot_files(root: str | Path) -> dict[str, float]:
    """Map project files → mtime, skipping the .falamus session-internals dir. Used to diff deliverables."""
    base = Path(root)
    out: dict[str, float] = {}
    if not base.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(base):
        if _SESSION_INTERNALS in dirnames:
            dirnames.remove(_SESSION_INTERNALS)     # don't descend into .falamus/
        for name in filenames:
            p = Path(dirpath) / name
            try:
                out[str(p.relative_to(base))] = p.stat().st_mtime
            except OSError:
                pass
    return out


def changed_since(root: str | Path, before: dict[str, float]) -> list[str]:
    """Project files that are new or whose mtime advanced since the `before` snapshot (the deliverables)."""
    after = snapshot_files(root)
    return sorted(rel for rel, mt in after.items() if mt > before.get(rel, -1.0))


# ── P2: policy — caller-settable vs server-fixed config, + an audit trail ───────────────────────────────
# A caller may tune BEHAVIOUR / QUALITY / COST per session; it may NOT change CAPABILITY / SECURITY — those
# stay from the SERVER's config, so an external caller cannot escalate what falamus may do to the machine.
_CALLER_SETTABLE = frozenset({
    "thinking", "auto_compact", "compact_threshold",
    "max_depth", "max_iters_main", "max_iters_sub",
    "read_chunk_chars", "repeat_penalty", "repeat_last_n",
})   # SENSITIVE, server-fixed: persistent_interactive_shell, allowed_paths, workdir, confirm_command/write, backend, model.
     # `lang` is intentionally NOT here: it's the human-UI language; a programmatic caller has no UI and the
     # model-facing text is always English, so it would do nothing.


def session_config(server_cfg: Any, overrides: dict[str, Any] | None = None) -> tuple[Any, list[str]]:
    """Build a per-session Config: copy the server's config, then apply ONLY caller-safe overrides. Sensitive
    or unknown keys are IGNORED (returned in `rejected`) so the caller can't escalate capability. Values go
    through the normal validated setter."""
    from dataclasses import replace

    from falamus.settings import apply_setting
    cfg = replace(server_cfg)
    rejected: list[str] = []
    for key, val in (overrides or {}).items():
        if key in _CALLER_SETTABLE:
            apply_setting(cfg, key, str(val))
        else:
            rejected.append(key)
    return cfg, rejected


class AuditLog:
    """Append-only JSONL record of a session's tool events + confirmation decisions — an in-app audit trail
    (pairs with out-of-band eBPF/tty monitoring). Best-effort: a write failure never breaks a run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, category: str, detail: dict[str, Any]) -> None:
        import json
        import time
        line = json.dumps({"ts": round(time.time(), 3), "cat": category, **detail},
                          ensure_ascii=False, default=str)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


class _BackendLike(Protocol):     # the slice of Backend that Session uses (so tests can supply a fake)
    workdir: str
    confirm_fn: Any
    event_sink: Any
    runtime: Any
    def run_message(self, text: str) -> str: ...
    def build(self, resume_sid: str | None = ...) -> None: ...


class Session:
    """A drivable falamus session. Wrap a built backend; each `run` is one multi-agent turn."""

    def __init__(
        self,
        backend: _BackendLike,
        on_confirm: Callable[[Any, dict, str], bool] | None = None,
        on_event: Callable[[str, str, Any], None] | None = None,
        audit: AuditLog | None = None,
        rejected_overrides: list[str] | None = None,
    ) -> None:
        self.backend = backend
        self._events: list[dict[str, Any]] = []
        self._on_event_cb = on_event
        self._audit = audit
        self.rejected_overrides = list(rejected_overrides or [])   # caller overrides that were NOT applied
        # no human at a terminal here: confirmations go to the supplied callback (default = auto-approve,
        # for trusted in-process use / testing). The destructive BLACKLIST still denies regardless.
        user_confirm = on_confirm or (lambda tool, args, reason: True)

        def _confirm(tool: Any, args: dict, reason: str) -> bool:
            approved = bool(user_confirm(tool, args, reason))
            if self._audit:
                self._audit.record("confirm", {"tool": getattr(tool, "name", str(tool)),
                                                "reason": reason, "approved": approved})
            return approved
        backend.confirm_fn = _confirm
        backend.event_sink = self._sink

    @classmethod
    def connect(
        cls,
        cfg: Any,
        model: str | None = None,
        overrides: dict[str, Any] | None = None,
        on_confirm: Callable[[Any, dict, str], bool] | None = None,
        on_event: Callable[[str, str, Any], None] | None = None,
        audit_path: str | Path | None = None,
    ) -> Session:
        """Build a real Backend and wrap it. `overrides` = caller-safe per-session settings (behaviour/quality/
        cost); sensitive/unknown ones are ignored (see `.rejected_overrides`). `audit_path` → a JSONL trail."""
        from falamus.core.backend import Backend
        cfg2, rejected = session_config(cfg, overrides)
        audit = AuditLog(audit_path) if audit_path else None
        return cls(Backend(cfg2, model=model), on_confirm=on_confirm, on_event=on_event,
                   audit=audit, rejected_overrides=rejected)

    def _sink(self, name: str, kind: str, data: Any) -> None:
        self._events.append({"agent": name, "kind": kind, "data": data})
        if self._audit:
            self._audit.record("event", {"agent": name, "kind": kind, "data": data})
        if self._on_event_cb:
            self._on_event_cb(name, kind, data)

    @property
    def sid(self) -> str:
        rt = self.backend.runtime
        return rt.session.sid if rt is not None else ""

    def run(self, task: str) -> RunResult:
        """Submit one task; run a full multi-agent turn; return a structured result."""
        self._events = []
        before = snapshot_files(self.backend.workdir)
        reply = self.backend.run_message(task)
        return RunResult(
            reply=reply,
            deliverables=changed_since(self.backend.workdir, before),
            status=classify_status(reply),
            sid=self.sid,
            events=list(self._events),
        )

    def resume(self, sid: str) -> None:
        """Reload a previous session by id (its conversation tail is replayed); later `run`s continue it."""
        self.backend.build(resume_sid=sid)

    def close(self) -> None:
        """Best-effort: close any persistent shell sessions (run_message already closes them per turn)."""
        rt = self.backend.runtime
        if rt is not None:
            rt.shell_mgr.close_all()
