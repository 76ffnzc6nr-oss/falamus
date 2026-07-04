"""Sequential agent chain: planner (main agent) + sequential sub-agents.

Design (sequential chain, not a parallel tree):
  - Sub-agents run one at a time (sequential); local models usually can't parallelize.
  - The main agent = planner/supervisor: first plans an ordered list of steps (= how many
    sub-agents to dispatch), then runs sub-agents "in order" to complete them step by step,
    verifying each in the shared workspace, finally mv/cp the results into the project.
  - Depth is not the main agent's concern: depth is purely a sub-agent's own judgment that
    "this step of mine is too complex and would pollute my own window", so it spawns its own
    sub-agent (the spawn tool is only attached when depth < max_depth).

Shared workspace: <session_dir>/work/ (shared by all agents; later ones see earlier outputs).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from falamus.persistence.rules import (
    inject_into_prompt,
    last_sub_rules,
    load_or_create_rules,
    read_rules,
    sub_rules,
    working_rules,
)
from falamus.persistence.session_store import SessionStore
from falamus.persistence.workspace import resolve_workdir, session_base
from falamus.prompt import frag
from falamus.tools import cli, files, image
from falamus.tools.registry import Tool, ToolRegistry, ToolResult
from falamus.tools.shell_session import SHELL_TOOLS, ShellManager, available, make_shell_tools

from .agent import Agent
from .client import LLMClient
from .context import ContextManager

# default tool sets for the main agent (light) and sub-agents (heavy tasks)
ORCH_TOOLS = ["read_file", "write_file", "append_file", "edit_file", "list_dir", "run_command", "view_image"]
SUB_TOOLS = ["read_file", "write_file", "append_file", "edit_file", "list_dir", "run_command", "view_image"]

RuntimeEventCb = Callable[[str, str, Any], None]  # (agent_name, kind, data)


def _is_windows() -> bool:
    """Isolated OS check (monkeypatch-friendly for tests; avoids patching os.name → pathlib WindowsPath)."""
    return os.name == "nt"


def _posix(p: object) -> str:
    """Forward-slash form for a path we EMBED in model-facing text. run_command runs through Git-Bash on
    Windows (see tools/cli.py), which wants '/', not '\\' (backslash = escape). The model pastes the paths
    we hand it straight into shell commands, so we hand them out POSIX-style. No-op on POSIX."""
    return str(p).replace("\\", "/") if _is_windows() else str(p)


# Windows-only system-prompt note: tell the model the truth (Windows + Git-Bash) so it keeps emitting unix
# commands + forward-slash paths instead of cmd verbs / backslashes that break under bash. Injected at
# runtime (like the headless note) ONLY on Windows → never touches the byte-identical POSIX prompts.
_WIN_ENV_NOTE = frag("notes", "win_env")


def _maybe_win_note(prompt: str) -> str:
    """Append the Windows env note to a system prompt when running on Windows; no-op on POSIX."""
    return prompt + "\n\n" + _WIN_ENV_NOTE if _is_windows() else prompt


def _last_text(messages: list, role: str) -> str:
    """Get the last plain-text content for a role (for the manifest's recent-exchange record)."""
    for m in reversed(messages):
        if m.get("role") == role and isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"].strip()
    return ""


_MAX_SPAWN_FAILS = 3   # dispatched sub-agents failing this many times "consecutively" (reset on a success) → circuit-break, escalate
# a sub-agent that returns these strings is treated as "failed" (not a normal deliver)
# A sub-agent's final string starting with one of these = the step did NOT complete (real stuck state).
# NOTE: the repeat-guard's "repeated an already-SUCCESSFUL call" stop is deliberately NOT here — that work
# WAS produced, so it's reported up as a normal (done) result telling the parent to verify & continue.
_FAIL_SENTINELS = ("[reached the max tool-iteration limit", "[interrupted]", "Sub-agent failed",
                   "[generation aborted: degenerate]", "[stopped: repeated a FAILING call")


# ──────────────────────────────────────────────────────────────────────────
# Session (agent states + shared workspace)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class AgentState:
    id: str
    role: str          # task purpose summary
    status: str = "running"   # running | done | error
    artifacts_dir: str = ""


@dataclass
class Session:
    root: Path
    sid: str
    _name_counts: dict = field(default_factory=dict)    # for naming: monotonic (no duplicate IDs)
    agents: dict[str, AgentState] = field(default_factory=dict)

    @classmethod
    def create(cls, base: str | Path, sid: str | None = None) -> Session:
        """Create a new session under base. base is usually <workdir>/.falamus/sessions."""
        sid = sid or time.strftime("%Y%m%d-%H%M%S")
        root = Path(base).expanduser() / sid
        (root / "work").mkdir(parents=True, exist_ok=True)   # workspace shared by all agents
        return cls(root=root, sid=sid)

    @property
    def work_dir(self) -> Path:
        """Workspace shared by all sub-agents (continuation: later ones see earlier outputs)."""
        d = self.root / "work"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _base_of(parent: str) -> str:
        return "sub" if parent == "main" else parent

    def next_child_id(self, parent: str) -> str:
        """Hierarchical naming: main's child=sub_1; sub_1's child=sub_1_1 … (monotonic, no reuse)."""
        base = self._base_of(parent)
        self._name_counts[base] = self._name_counts.get(base, 0) + 1
        return f"{base}_{self._name_counts[base]}"



# ──────────────────────────────────────────────────────────────────────────
# Runtime: builds the main agent, executes spawn
# ──────────────────────────────────────────────────────────────────────────
class AgentRuntime:
    def __init__(
        self,
        client: LLMClient,
        session: Session,
        *,
        workdir: str | None = None,
        max_depth: int = 2,
        main_max_iters: int = 0,
        sub_max_iters: int = 60,
        read_chunk_chars: int = 8000,
        on_event: RuntimeEventCb | None = None,
        guard: Any = None,
        auto_compact: bool = True,
        compact_threshold: float = 0.7,
        lang: str = "en",
        cancel_check: Any = None,
        error_log: Any = None,
    ) -> None:
        self.client = client
        self.session = session
        self.workdir = str(resolve_workdir(workdir))   # project root (the working directory)
        self.max_depth = max_depth   # sequential execution → only limit depth (prevent infinite recursion), not width
        self.main_max_iters = main_max_iters   # 0 = unlimited (main's guard is the circuit breaker)
        self.sub_max_iters = sub_max_iters     # finite backstop for sub-agents
        self.read_chunk_chars = read_chunk_chars  # suggested read-chunk size passed to the file tools
        self.on_event = on_event
        self.guard = guard                  # safety guard (applied to every tool registry)
        self.auto_compact = auto_compact
        self.compact_threshold = compact_threshold
        self.lang = lang
        self.cancel_check = cancel_check
        self.error_log = error_log
        self.store = SessionStore(session.root)
        self.resuming = False          # resume mode: build_orchestrator will restore the main agent
        self.force_plain_chat = False  # set by Backend when no working shell → run tool-less (plain chat)
        self.enable_shell = False      # opt-in (--enable-shell): persistent interactive shell-session tools
        self.shell_mgr = ShellManager()  # holds this turn's live shell sessions (closed at turn end by Backend)
        self.mcp_clients: list = []      # external MCP servers (client direction): session-scoped, closed by Backend
        self._mcp_tools_cache: list | None = None  # bridged external tools, discovered once, shared across agents
        self._title = ""               # session title (first user message); per-instance, not shared
        self._recent: dict = {}        # most recent exchange snippet; per-instance, not shared
        self._save_manifest()

    @classmethod
    def start(
        cls,
        client: LLMClient,
        workdir: str | None = None,
        *,
        max_depth: int = 2,
        on_event: RuntimeEventCb | None = None,
        **kw: Any,
    ) -> AgentRuntime:
        """Convenience constructor: anchored at the project root, the session is created under <workdir>/.falamus/sessions."""
        wd = resolve_workdir(workdir)
        session = Session.create(base=session_base(wd))
        return cls(client, session, workdir=str(wd), max_depth=max_depth, on_event=on_event, **kw)

    @classmethod
    def resume(
        cls,
        client: LLMClient,
        sid: str,
        workdir: str | None = None,
        *,
        max_depth: int = 2,
        on_event: RuntimeEventCb | None = None,
        **kw: Any,
    ) -> AgentRuntime:
        """Restore an existing session: rebuild the runtime; build_orchestrator reloads the main agent conversation."""
        wd = resolve_workdir(workdir)
        root = session_base(wd) / sid
        if not root.exists():
            raise FileNotFoundError(f"session not found: {root}")
        session = Session(root=root, sid=sid)
        rt = cls(client, session, workdir=str(wd), max_depth=max_depth, on_event=on_event, **kw)
        rt.resuming = True
        # keep the original session title (so the rebuilt manifest doesn't clear it)
        rt._title = rt.store.load_manifest().get("title", "")
        rt._save_manifest()
        return rt

    # ---- manifest / checkpoints ----------------------------------------
    def _save_manifest(self) -> None:
        info = self.client.info
        self.store.save_manifest({
            "sid": self.session.sid,
            "workdir": self.workdir,
            "backend": getattr(info, "backend", ""),
            "base_url": getattr(info, "base_url", ""),
            "model": getattr(info, "name", ""),
            "title": self._title,
            "recent": self._recent,        # the most recent exchange (know what was discussed without reading the full checkpoint)
            "agents": [vars(s) for s in self.session.agents.values()],
        })

    def _checkpoint_cb(self, name: str):
        def cb(agent_name: str, messages: list[dict[str, Any]]) -> None:
            self.store.save_agent(agent_name, messages)
            if agent_name != "main":
                return
            # main agent: title (first message, set once) + recent-exchange snippet (updated each time, no model call)
            if not self._title:
                first = next((m.get("content") for m in messages
                              if m.get("role") == "user" and isinstance(m.get("content"), str)), "")
                if first:
                    self._title = first[:40]
            self._recent = {
                "last_user": _last_text(messages, "user")[:300],
                "last_assistant": _last_text(messages, "assistant")[:300],
            }
            self._save_manifest()
        return cb

    # ---- events ---------------------------------------------------------
    def _agent_cb(self, name: str) -> Callable[[str, Any], None]:
        def cb(kind: str, data: Any) -> None:
            if self.on_event:
                self.on_event(name, kind, data)
        return cb

    # ---- main agent -----------------------------------------------------
    def build_orchestrator(self) -> Agent:
        # SOLO = main works alone, NO sub-agents: max_depth <= 0 (subagents disabled by config). The spawn
        # tool is then never attached and the persona/notes/turn-reminder switch to the solo variants.
        solo = self.max_depth <= 0
        reg = self._base_registry()
        if not solo:
            reg.register(self._make_spawn_tool(depth=1, parent_name="main"))
        reg.register(_make_memo_tool(self.store, "main"))
        allowed = (ORCH_TOOLS + (["memo"] if solo else ["spawn_subagent", "memo"])
                   + self._shell_names() + self._mcp_names())
        native = bool(self.client.info and self.client.info.supports_tools) and not self.force_plain_chat
        if not native:
            # Plain chat: send NO tools. Do NOT CREATE falamus.md (it's the tool-version template), but DO
            # inject one if it ALREADY EXISTS (user-authored rules and/or a saved progress summary). System
            # message is otherwise blank. _chat_only runs it.
            if self.on_event:
                reason = "no usable shell" if self.force_plain_chat else "model does not support tools"
                self.on_event("main", "no_tools", f"{reason} — plain chat (no tools)")
            existing = read_rules(self.workdir).strip()
            rules_block = inject_into_prompt("", existing).strip() if existing else ""
            system_prompt = rules_block
            turn_reminder = ""
        else:
            # inject falamus.md rules
            rules_text, created = load_or_create_rules(self.workdir)
            if self.on_event:
                self.on_event("main", "rules", "created falamus.md template" if created else "loaded falamus.md")
            persona = frag("agents", "orchestrator_solo") if solo else frag("agents", "orchestrator")
            system_prompt = inject_into_prompt(persona, rules_text)
            if not solo:
                # the staging→project workspace only matters when sub-agents stage files there
                system_prompt += "\n\n" + frag("notes", "workspace").format(
                    work=_posix(self.session.work_dir), workdir=_posix(self.workdir),
                )
            system_prompt = _maybe_win_note(system_prompt)
            if solo:
                # no sub-agents → the "you work alone" note replaces the depth budget
                system_prompt += "\n\n" + frag("notes", "solo")
            else:
                # main is depth 0 → it may delegate self.max_depth levels below it
                system_prompt += "\n\n" + frag("notes", "depth_budget").format(n=self.max_depth)
                system_prompt += "\n\n" + frag("notes", "iter_budget_main").format(n=self.sub_max_iters)
            reminder_frag = "turn_reminder_solo" if solo else "turn_reminder"
            turn_reminder = frag("notes", reminder_frag).format(tools=", ".join(allowed))
        orch = Agent(
            self.client, reg,
            system_prompt=system_prompt,
            allowed_tools=allowed,
            name="main",
            plain_chat=self.force_plain_chat,   # no working shell → tool-less plain chat
            max_iters=self.main_max_iters,   # 0 = unlimited; wide dispatch easily exceeds a small cap
            max_tokens=-1,                # main agent output unbounded (until EOS)
            turn_reminder=turn_reminder,
            on_event=self._agent_cb("main"),
            context_manager=self._make_cm("main"),
            checkpoint_cb=self._checkpoint_cb("main"),
            cancel_check=self.cancel_check,
            error_log=self.error_log,
        )
        # resume: reload the main agent's conversation history
        if self.resuming:
            saved = self.store.load_agent("main")
            if saved:
                orch.restore(saved)
                if self.on_event:
                    self.on_event("main", "resume", f"restored {len(saved)} messages")
        return orch

    def _make_cm(self, name: str) -> ContextManager | None:
        if not self.auto_compact:
            return None
        n_ctx = self.client.info.n_ctx if self.client.info else 8192
        return ContextManager(
            self.client, n_ctx, threshold=self.compact_threshold,
            on_event=lambda k, d: self.on_event(name, k, d) if self.on_event else None,
        )

    def _base_registry(self, root: str | None = None, owner: str = "main") -> ToolRegistry:
        """Tool registry; root is the base dir for file/CLI tools (main agent=project root; sub-agent=shared workspace).
        owner names the agent for per-agent shell-session ownership."""
        root = root or self.workdir
        reg = ToolRegistry()
        reg.register_all(files.make_tools(root, read_chunk=self.read_chunk_chars))
        reg.register_all(cli.make_tools(root))
        reg.register_all(image.make_tools(root))
        if self.enable_shell and available():
            reg.register_all(make_shell_tools(self.shell_mgr, owner, root))
        reg.register_all(self._mcp_tools())     # external MCP tools (namespaced), shared across agents
        reg.guard = self.guard
        return reg

    def _shell_names(self) -> list[str]:
        """Shell-session tool names to add to an allowlist when the feature is enabled (else empty)."""
        return list(SHELL_TOOLS) if (self.enable_shell and available()) else []

    # ---- external MCP tools (client direction) --------------------------
    def _mcp_tools(self) -> list:
        """Bridged tools from the configured external MCP servers — discovered ONCE (connecting lazily on the
        first registry build), then shared across every agent. Empty if none are configured."""
        if self._mcp_tools_cache is None:
            self._mcp_tools_cache = self._discover_mcp()
        return self._mcp_tools_cache

    def _discover_mcp(self) -> list:
        from falamus.mcp_bridge import bridged_tools
        from falamus.mcp_client import McpClient, McpError
        from falamus.mcp_config import load_mcp_servers, server_command
        tools: list = []
        for name, spec in load_mcp_servers().items():
            client = McpClient(server_command(spec), name=name)
            try:
                client.start()
                tools.extend(bridged_tools(client))
                self.mcp_clients.append(client)
                if self.on_event:
                    self.on_event("main", "mcp", f"connected external MCP '{name}' ({len(client.tools)} tools)")
            except McpError as e:               # a down/broken server is skipped, not fatal to the session
                client.close()
                if self.on_event:
                    self.on_event("main", "mcp_error", f"external MCP '{name}': {e}")
        return tools

    def _mcp_names(self) -> list[str]:
        return [t.name for t in self._mcp_tools()]

    def close_mcp(self) -> None:
        for c in self.mcp_clients:
            c.close()
        self.mcp_clients = []

    # ---- spawn tool -----------------------------------------------------
    def _make_spawn_tool(self, depth: int, parent_name: str) -> Tool:
        cb: dict[str, Any] = {"fails": 0, "failed_tasks": []}   # this parent's circuit-breaker state

        def handler(args: dict) -> ToolResult:
            task = args.get("task", "")
            if not task:
                return ToolResult.error("spawn_subagent requires a 'task' argument")
            norm = " ".join(task.split())[:160].lower()
            # hard circuit-break: sub-agents here have failed up to the limit → refuse to spawn, force escalation
            if cb["fails"] >= _MAX_SPAWN_FAILS:
                return ToolResult.error(
                    f"Sub-agents here have failed {cb['fails']} times in a row (no progress), at the limit. Stop "
                    "spawning: use deliver to write a FAILURE summary (where it's stuck, what you tried, what's "
                    "done, how to split it) and report up for re-planning. Do not spawn more sub-agents."
                )
            # same task just failed → block to avoid a verbatim-respawn death loop
            if norm in cb["failed_tasks"]:
                return ToolResult.error(
                    "This task was just dispatched and failed. Do NOT re-spawn it verbatim — use a smaller/"
                    "different breakdown, or report up for re-planning."
                )
            ctx = args.get("context_hint", "")
            allowed = args.get("allowed_tools")
            mt = args.get("max_tokens", -1)        # output unbounded by default
            res = self.spawn(task, ctx, allowed, depth, parent_name, mt, args.get("output_name"))

            if res.get("failed"):
                cb["fails"] += 1
                cb["failed_tasks"].append(norm)
                if self.error_log:
                    self.error_log.log(
                        parent_name, "spawn_failed",
                        f"child {res['agent_id']} failed (#{cb['fails']}/{_MAX_SPAWN_FAILS}) task: {task[:120]}",
                        res.get("summary", ""),
                    )
                tail = ("(failure limit reached; stop spawning, deliver a failure summary and report up for re-planning)"
                        if cb["fails"] >= _MAX_SPAWN_FAILS else
                        "Diagnose the cause first; if the task is too complex, split it into smaller steps; never re-spawn verbatim.")
                done = ", ".join(res.get("files") or []) or "(none)"
                return ToolResult.error(
                    f"Sub-agent {res['agent_id']} failed (reason: {(res['summary'] or 'no summary')[:200]}). "
                    f"Consecutive failure {cb['fails']}/{_MAX_SPAWN_FAILS} here. {tail}\n"
                    f"! The workspace already has files (parts completed before the failure — don't redo, only fill gaps): {done}\n"
                    f"Workspace: {res.get('artifacts_dir', '')}"
                )

            # success → reset the consecutive-failure counter (only "consecutive" failures circuit-break; progress counts)
            cb["fails"] = 0
            extra = res.get("extra_paths") or []
            extra_line = f"Produced elsewhere: {', '.join(extra)}\n" if extra else ""
            names = res.get("output_names") or []
            out_line = (f"This step's deliverable file(s): {', '.join(names)} "
                        f"(collect/move THESE exact files by name).\n"
                        if names else "")
            return ToolResult(text=(
                f"Sub-agent {res['agent_id']} done.\n"
                f"Summary: {res['summary']}\n"
                f"Shared workspace: {res['artifacts_dir']}\n"
                f"Workspace files: {', '.join(res['files']) or '(none)'}\n"
                f"{out_line}"
                f"{extra_line}"
                f"! You are the dispatcher — verify: confirm the above files exist and are complete "
                f"(read_file to spot-check, run_command `wc -c` vs the source length); if incomplete, re-dispatch/split smaller.\n"
                f"(Use read_file / list_dir on the above paths for details.)"
            ))

        return Tool(
            name="spawn_subagent",
            description=(
                frag("tools", "spawn_subagent")
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "the clear goal for the sub-agent to complete"},
                    "context_hint": {"type": "string", "description": "necessary background to pass to the sub-agent (only what's needed)"},
                    "output_name": {
                        "type": "string",
                        "description": "OPTIONAL. The output filename(s) this step should produce — comma-separate "
                                       "multiple names (e.g. `a.md, b.md`). Use clear, DISTINCT names (different "
                                       "deliverables → different names). The sub-agent writes EXACTLY these files (used "
                                       "as-is, never renamed) and reports them back so you collect the right ones. If it "
                                       "re-delegates, it passes the SAME name(s) down, so each file is found by its name "
                                       "however many layers deep it is produced.",
                    },
                    "allowed_tools": {
                        "type": "array", "items": {"type": "string"},
                        "description": "restrict the sub-agent's available tool names (omit to use the default set)",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "the sub-agent's per-output cap; -1=unbounded (for writing/large code), omit=default limit (good for tests, etc.)",
                    },
                },
                "required": ["task"],
            },
            handler=handler,
            risk="medium",
        )

    # ---- run a single sub-agent -----------------------------------------
    def spawn(
        self,
        task: str,
        context_hint: str = "",
        allowed_tools: list[str] | None = None,
        depth: int = 1,
        parent_name: str = "main",
        max_tokens: int | None = -1,        # output unbounded by default
        output_name: str | None = None,     # the deliverable filename this step must produce (made unique here)
    ) -> dict[str, Any]:
        agent_id = self.session.next_child_id(parent_name)
        work = self.session.work_dir          # workspace shared by all agents (continuation: they see each other's outputs)
        state = AgentState(id=agent_id, role=task[:200], artifacts_dir=str(work))
        self.session.agents[agent_id] = state
        self._save_manifest()
        if self.on_event:
            self.on_event(agent_id, "spawn", state)

        # sub-agent tools: root = shared workspace (files written here are reachable by the main agent and other sub-agents)
        delivery: dict[str, Any] = {"summary": "", "files": [], "extra_paths": []}
        reg = self._base_registry(root=str(work), owner=agent_id)
        reg.register(_make_deliver_tool(work, agent_id, delivery))
        # explicit allowed_tools from the parent wins; the default set gets the shell tools (when enabled)
        names = list(allowed_tools) if allowed_tools else list(SUB_TOOLS) + self._shell_names() + self._mcp_names()
        names += ["deliver"]
        can_spawn = depth < self.max_depth
        if can_spawn:
            # only non-leaf (coordinating) sub-agents get spawn + memo: a leaf sub is single-shot
            # (deliver-and-die), so a private memo has no later reader. (main always has memo too.)
            reg.register(self._make_spawn_tool(depth + 1, parent_name=agent_id))
            reg.register(_make_memo_tool(self.store, agent_id))
            names += ["spawn_subagent", "memo"]
        # the depth self-assessment note is only given to sub-agents that can actually spawn
        sub_prompt = frag("agents", "sub").format(work=_posix(work), workdir=_posix(self.workdir))
        if can_spawn:
            sub_prompt += "\n\n" + frag("agents", "sub_depth")
        # Working rules per role: a sub that can still spawn delegates AND does work (A+C); a leaf sub that
        # cannot spawn only does work (A+D). Single source — COMMON (A) is shared with the main's falamus.md.
        sub_prompt += "\n\n" + working_rules(sub_rules() if can_spawn else last_sub_rules())
        sub_prompt = _maybe_win_note(sub_prompt)
        # tell the sub how many more levels it may still spawn below itself (0 = leaf → do it yourself)
        sub_prompt += "\n\n" + frag("notes", "depth_budget").format(n=self.max_depth - depth)
        sub_prompt += "\n\n" + frag("notes", "iter_budget").format(n=self.sub_max_iters)

        sub = Agent(
            self.client, reg,
            system_prompt=sub_prompt,
            allowed_tools=names,
            name=agent_id,
            max_iters=self.sub_max_iters,
            max_tokens=max_tokens,
            on_event=self._agent_cb(agent_id),
            context_manager=self._make_cm(agent_id),
            checkpoint_cb=self._checkpoint_cb(agent_id),
            cancel_check=self.cancel_check,
            error_log=self.error_log,
        )
        prompt = task if not context_hint else f"Context: {context_hint}\n\nTask: {task}"
        # use the orchestrator's chosen name AS-IS (no auto-suffixing): the filename IS the deliverable's
        # identity. Same name = same file across any number of re-delegation layers, so the dispatcher and the
        # sub-agent always agree on where the file is. (Distinct deliverables must be given distinct names by
        # the model — auto-suffixing silently renamed files and broke that shared expectation; see test13.)
        # A step may declare ONE name or SEVERAL (comma-separated) — purely to pass the target filenames down;
        # nothing validates them. Distinct deliverables must be given distinct names by the model (see test13).
        out_names = [n.strip().lstrip("/") for n in (output_name or "").replace("\n", ",").split(",") if n.strip()]
        if len(out_names) == 1:
            prompt += "\n\n" + frag("notes", "output_file_one").format(name=out_names[0])
        elif out_names:
            listed = ", ".join(f"`{n}`" for n in out_names)
            prompt += "\n\n" + frag("notes", "output_file_many").format(listed=listed)
        try:
            final = sub.run(prompt)
            state.status = "done"
        except Exception as e:  # noqa: BLE001
            state.status = "error"
            final = f"Sub-agent failed: {e}"

        summary = delivery["summary"] or final
        # failure decision: raised an exception / hit the iteration limit / was interrupted (these are the real stuck states)
        failed = (
            state.status == "error"
            or any(str(final).startswith(s) for s in _FAIL_SENTINELS)
        )
        if failed and state.status != "error":
            state.status = "error"
        # list shared-workspace files from disk (ground truth)
        on_disk = sorted(p.name for p in work.iterdir() if p.is_file())
        if failed:
            # the sub-agent had no chance to deliver: auto-generate a "progress + error" report (persisted to the workspace and returned up as the summary)
            done = ", ".join(on_disk) or "(none)"
            report = (
                f"# {agent_id} — FAILED\n"
                f"Task: {task[:300]}\n"
                f"Error: {final}\n"
                f"Files completed before the failure (reusable): {done}\n"
                f"Suggestion: the parent should read the workspace and reuse these files, only filling the gaps.\n"
            )
            try:
                (work / f"SUMMARY_{agent_id}.md").write_text(report, encoding="utf-8")
            except OSError:
                pass
            summary = f"FAILED — error: {str(final)[:200]}; completed before failure: {done}"
        self._save_manifest()
        if self.on_event:
            self.on_event(agent_id, "done", state)
        return {
            "agent_id": agent_id,
            "summary": summary,
            "artifacts_dir": str(work),
            "files": on_disk,
            "extra_paths": delivery.get("extra_paths", []),
            "failed": failed,
            "output_names": out_names,       # the deliverable name(s) this step was told to write (possibly [])
        }


# ──────────────────────────────────────────────────────────────────────────
# memo tool: the agent's own scratchpad / checklist, stored EXTERNALLY (per agent)
# ──────────────────────────────────────────────────────────────────────────
def _make_memo_tool(store: SessionStore, agent_name: str) -> Tool:
    """The agent's private to-do list, kept in an external per-agent file (NOT injected into the prompt, so
    the KV cache stays valid). Call with no content to READ it; call with `content` to OVERWRITE it."""
    def handler(args: dict) -> ToolResult:
        content = args.get("content")
        if content is not None:                       # write mode: overwrite the memo
            store.save_memo(agent_name, content)
            return ToolResult(text="Memo updated (saved to your private external store).")
        cur = store.load_memo(agent_name)             # read mode: return the current memo
        return ToolResult(text=(f"Your current memo:\n{cur}" if cur.strip()
                                else "Your memo is empty — no to-do items recorded yet."))

    return Tool(
        name="memo",
        description=frag("tools", "memo"),
        parameters={
            "type": "object",
            "properties": {"content": {"type": "string",
                "description": "the full updated memo text (OMIT entirely to READ the current memo instead of writing)"}},
            # content is optional: omitting it = read; providing it = overwrite
        },
        handler=handler,
        risk="low",
    )


# ──────────────────────────────────────────────────────────────────────────
# deliver tool: report a summary (outputs already live in the shared workspace)
# ──────────────────────────────────────────────────────────────────────────
def _make_deliver_tool(work: Path, agent_id: str, delivery: dict[str, Any]) -> Tool:
    def handler(args: dict) -> ToolResult:
        summary = args.get("summary", "")
        filename = args.get("filename")
        content = args.get("content")
        paths = args.get("paths") or []
        delivery["summary"] = summary
        written = ""
        if filename and content is not None:
            p = work / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            delivery["files"].append(filename)
            written = f", and wrote {filename}"
        if isinstance(paths, list):
            delivery["extra_paths"] = [str(x) for x in paths]   # file paths produced elsewhere
        # one summary per sub-agent (don't overwrite each other)
        (work / f"SUMMARY_{agent_id}.md").write_text(summary, encoding="utf-8")
        return ToolResult(text=f"Delivered{written}. End the task now.")

    return Tool(
        name="deliver",
        description=frag("tools", "deliver"),
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "a concise result summary for the parent agent"},
                "filename": {"type": "string", "description": "filename to write into the shared workspace (optional, e.g. result.json)"},
                "content": {"type": "string", "description": "the full content of that file (optional)"},
                "paths": {
                    "type": "array", "items": {"type": "string"},
                    "description": "absolute paths of result files you wrote outside the shared workspace (optional)",
                },
            },
            "required": ["summary"],
        },
        handler=handler,
        risk="low",
    )


# Prompts (personas) live in the active set's agents.md and are read at RUNTIME via frag() (see
# build_orchestrator / spawn) so the active set — local vs cloud — is honoured. Reply language is NOT
# forced here — it follows the user's input language (UI language is i18n, unrelated to this).


if __name__ == "__main__":
    import sys

    client = LLMClient("http://localhost:8080")
    client.detect()

    def log(name: str, kind: str, data: Any) -> None:
        if kind == "tool_call":
            print(f"  [{name}] call {data.name}({data.arguments})")
        elif kind == "spawn":
            print(f"  [{name}] > spawning sub-agent: {data.role}")
        elif kind == "done":
            print(f"  [{name}] ok sub-agent finished: {data.status}")
        elif kind == "final":
            print(f"  [{name}] * {str(data)[:120]}")

    rt = AgentRuntime.start(client, workdir="/home/c/helper", on_event=log)
    print("session:", rt.session.root)

    mode = sys.argv[1] if len(sys.argv) > 1 else "direct"
    if mode == "direct":
        print("\n=== [A] directly test the runtime.spawn() contract ===")
        res = rt.spawn(
            task="Count how many .py files are under /home/c/helper (excluding the .venv dir), list their "
                 "names, and write the result into result.txt and deliver it. Hint: you can use "
                 "find /home/c/helper -path '*/.venv' -prune -o -name '*.py' -print",
        )
        print("\nspawn returned:")
        print("  agent_id :", res["agent_id"])
        print("  summary  :", res["summary"][:200])
        print("  artifacts:", res["artifacts_dir"])
        print("  files    :", res["files"])
    else:
        print("\n=== [B] end-to-end: the main agent dispatches on its own ===")
        orch = rt.build_orchestrator()
        out = orch.run("Count how many .py files this project has and list their names. This is exploration work; please dispatch a sub-agent.")
        print("\nmain agent reply:", out)
