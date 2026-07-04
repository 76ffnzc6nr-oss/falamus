"""Multi-agent UI status bar.

Consumes the runtime's events `on_event(agent_name, kind, data)`, tracks each agent's state,
and renders "how many agents are running, each one's purpose summary and current action" into a
live status panel.

  - StatusTracker: pure state (unit-testable).
  - render_panel(): draw the state into a multi-line string.
  - LiveStatus: wires the tracker to events and refreshes in place in the terminal (ANSI).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, TextIO

_SYM = {"running": ">", "done": "ok", "error": "x", "idle": "*"}


@dataclass
class AgentRow:
    id: str
    label: str                 # "main" / "sub"
    task: str = ""             # purpose summary
    status: str = "running"    # running | done | error | idle
    steps: int = 0
    last_action: str = ""


@dataclass
class StatusTracker:
    rows: dict[str, AgentRow] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def _ensure(self, name: str) -> AgentRow:
        if name not in self.rows:
            is_main = name == "main"
            self.rows[name] = AgentRow(
                id=name,
                label="main" if is_main else "sub",
                task="orchestrate / plan" if is_main else "",
                status="idle" if is_main else "running",
            )
            self.order.append(name)
        return self.rows[name]

    def update(self, name: str, kind: str, data: Any) -> None:
        row = self._ensure(name)
        if kind == "spawn":
            # data is an AgentState (has .role)
            row.task = getattr(data, "role", row.task)
            row.status = "running"
        elif kind == "tool_call":
            row.steps += 1
            row.status = "running"
            args = getattr(data, "arguments", {}) or {}
            brief = next((str(v) for v in args.values()), "")
            row.last_action = f"{getattr(data, 'name', '?')} {brief}".strip()[:48]
        elif kind == "done":
            row.status = getattr(data, "status", "done")
            row.last_action = ""
        elif kind == "final":
            if row.label == "main":
                row.status = "idle"
                row.last_action = ""
        elif kind == "compact":
            row.last_action = f"compacting context… {str(data)[:30]}"

    # ---- counts ---------------------------------------------------------
    def counts(self) -> tuple[int, int]:
        active = sum(1 for r in self.rows.values()
                     if r.label == "sub" and r.status == "running")
        done = sum(1 for r in self.rows.values()
                   if r.label == "sub" and r.status in ("done", "error"))
        return active, done


def render_panel(tracker: StatusTracker) -> str:
    subs = [tracker.rows[n] for n in tracker.order if tracker.rows[n].label != "main"]
    lines: list[str] = []
    main = tracker.rows.get("main")
    if main:
        tail = f"  {main.last_action}" if main.last_action else "  idle"
        lines.append(f"{_SYM.get(main.status, '*')} main{tail}")
    for i, r in enumerate(subs):
        branch = "└─" if i == len(subs) - 1 else "├─"
        sym = _SYM.get(r.status, ">")
        task = (r.task or "")[:28].ljust(28)
        if r.status == "running":
            state = f"running ({r.steps} steps)"
            if r.last_action:
                state += f" · {r.last_action}"
        elif r.status == "done":
            state = "done"
        elif r.status == "error":
            state = "error"
        else:
            state = r.status
        lines.append(f"{branch} {sym} #{r.id} {task} {state}")
    active, done = tracker.counts()
    lines.append(f"active agents: {active} / done: {done}")
    return "\n".join(lines)


def render_oneline(tracker: StatusTracker) -> str:
    """One-line summary (for the bottom status bar)."""
    active, done = tracker.counts()
    cur = next((tracker.rows[n] for n in tracker.order
                if tracker.rows[n].label == "sub" and tracker.rows[n].status == "running"), None)
    main = tracker.rows.get("main")
    if cur:
        head = f"> #{cur.id} {cur.task[:16]} ({cur.steps} steps)"
        if cur.last_action:
            head += f" {cur.last_action[:24]}"
    elif main and main.last_action:
        head = f"* {main.last_action[:30]}"
    else:
        head = "* idle"
    return f"{head}  │  active: {active} done: {done}"


class LiveStatus:
    """Wire the tracker to events and refresh the panel in place in the terminal.

    Usage:
        live = LiveStatus()
        rt = AgentRuntime.start(client, workdir, on_event=live.on_event)
        ...
    On non-TTY (or enabled=False) it degrades to no refresh (state still updates, can render later).
    """

    def __init__(self, stream: TextIO | None = None, enabled: bool | None = None) -> None:
        self.tracker = StatusTracker()
        self.stream = stream or sys.stdout
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self._lines = 0

    def on_event(self, name: str, kind: str, data: Any) -> None:
        self.tracker.update(name, kind, data)
        if self.enabled:
            self.refresh()

    def refresh(self) -> None:
        panel = render_panel(self.tracker)
        n = panel.count("\n") + 1
        # go back to the previous panel start and clear the old content
        if self._lines:
            self.stream.write(f"\033[{self._lines}A")
        for line in panel.split("\n"):
            self.stream.write("\033[2K" + line + "\n")
        self.stream.flush()
        self._lines = n

    def freeze(self) -> None:
        """Freeze the current panel on exit (later output continues below)."""
        if self.enabled:
            self.refresh()
        self._lines = 0


if __name__ == "__main__":

    # simulate events to demo panel rendering (visible even on non-TTY)
    class _S:
        def __init__(self, role, status="running"):
            self.role = role
            self.status = status

    class _C:
        def __init__(self, name, **a):
            self.name = name
            self.arguments = a

    t = StatusTracker()
    t.update("main", "tool_call", _C("spawn_subagent", task="look up three quotes"))
    t.update("a1", "spawn", _S("look up quotes A/B/C"))
    for _ in range(12):
        t.update("a1", "tool_call", _C("run_command", command="curl ..."))
    t.update("a2", "spawn", _S("aggregate the csv files under reports"))
    t.update("a2", "tool_call", _C("list_dir", path="reports"))
    t.update("a3", "spawn", _S("install dependencies"))
    t.update("a3", "done", _S("install dependencies", "done"))
    t.update("main", "final", "done")
    print(render_panel(t))
