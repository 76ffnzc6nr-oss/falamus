"""Backend controller: decouples LLM/runtime/command logic from the UI.

The UI (TUI or plain-text fallback) only needs to:
  - set backend.event_sink = fn(name, kind, data)  to receive agent events
  - set backend.confirm_fn = fn(tool, args, reason)->bool  to handle dangerous-action confirmation
  - call backend.run_message(text)  run on a background thread (blocking, so the UI starts a thread)
  - call backend.command(line, out)  to handle / commands
  - backend.request_cancel()  to interrupt the current task (ESC)
"""

from __future__ import annotations

import atexit
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import falamus.i18n as i18n
from falamus.core import providers, secrets
from falamus.core.client import BackendError, LLMClient
from falamus.core.safety import SafetyPolicy, make_guard
from falamus.core.subagent import AgentRuntime
from falamus.persistence.error_log import ErrorLog
from falamus.persistence.rules import update_last_progress
from falamus.persistence.session_store import list_sessions
from falamus.prompt import frag
from falamus.settings import PROGRAM_ROOT, SETTABLE, Config, apply_setting
from falamus.version import __version__

if TYPE_CHECKING:
    from falamus.core.agent import Agent

def _summary_prompt() -> str:
    """The /exit conversation-summary instruction (read at runtime → honours the active prompt set)."""
    return frag("notes", "summary") + "\n\n"


def is_in_program(path: Path) -> bool:
    p = path.resolve()
    return p == PROGRAM_ROOT or PROGRAM_ROOT in p.parents


class Backend:
    def __init__(self, cfg: Config, force_plain_chat: bool = False,
                 model: str | None = None) -> None:
        self.cfg = cfg
        self.force_plain_chat = force_plain_chat   # no working shell → run tool-less (plain chat) by choice
        # persistent interactive shell tools: gated by config (default off).
        # Set [tools] persistent_interactive_shell = true (or --persistent-interactive-shell true). POSIX only.
        self.enable_shell = bool(cfg.persistent_interactive_shell)
        # cloud provider (anthropic/…) vs local: cloud uses the registry endpoint + a stored api-key;
        # local keeps the existing auto-detect-from-base_url path. See providers.py / secrets.py.
        c_backend, base_url, api_key, c_model = self._resolve_backend(cfg, model)
        self.client = LLMClient(
            base_url,
            model=c_model,
            backend=c_backend,
            api_key=api_key,
            default_max_tokens=cfg.max_tokens,
            enable_thinking=(cfg.thinking.lower() != "off"),   # thinking switch (low/medium/high → on)
            repeat_penalty=cfg.repeat_penalty, repeat_last_n=cfg.repeat_last_n,
        )
        self.info = self.client.detect()
        self.workdir = cfg.workdir
        self.cancel_event = threading.Event()
        self.event_sink: Callable[[str, str, object], None] | None = None
        self.confirm_fn: Callable[[object, dict, str], bool] | None = None
        self.dev_mode = False
        self._policy = None
        self.error_log = ErrorLog.for_workdir(self.workdir)   # build() refreshes it for the current workdir
        self.server_online = True
        self._nctx_refreshed = False    # ollama: re-read the real loaded n_ctx after the first chat warms the model
        self.runtime: AgentRuntime | None = None
        self.orch: Agent | None = None
        self.build()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        # safety net: on exit (crash/Ctrl-C) kill any live shell session + close external MCP connections
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        if self.runtime:
            self.runtime.shell_mgr.close_all()
            self.runtime.close_mcp()

    @staticmethod
    def _resolve_backend(cfg: Config, model: str | None) -> tuple[str, str, str | None, str | None]:
        """Decide (backend, base_url, api_key, model) for the LLMClient. Cloud provider → registry endpoint
        + stored key (deterministic; the interactive picker is a later increment); local → auto-detect."""
        if providers.is_cloud(cfg.backend):
            prov = providers.get(cfg.backend)
            assert prov is not None
            try:
                key = secrets.load_api_key(cfg.backend)
            except secrets.SecretsUnavailable as e:
                raise BackendError(str(e)) from e
            if not key:
                raise BackendError(
                    f"no API key stored for {cfg.backend}. Set it with: falamus --set-key {cfg.backend} <key>")
            base_url = cfg.base_url if cfg.base_url.startswith("http") and "anthropic" in cfg.base_url else prov.endpoint
            return cfg.backend, base_url, key, (model or cfg.model or prov.default_model)
        return "auto", cfg.base_url, None, (model or None)

    # ---- heartbeat: detect server online/crashed (replaces a fixed timeout) ----
    def _heartbeat(self) -> None:
        fails = 0
        while True:
            ok = self.client.ping()
            self.server_online = ok      # status bar shows offline within one interval (~15s)
            if not ok:
                fails += 1
                # ~1 min of consecutive losses → treat as crashed, cancel the running task (avoid hanging).
                # (More fails needed than before so a single slow/timed-out ping doesn't false-cancel.)
                if fails >= 4 and not self.cancel_event.is_set():
                    self.cancel_event.set()
                    self.error_log.log("server", "offline", "heartbeat failed; server down/crashed → cancel")
            else:
                fails = 0
            time.sleep(15)   # heartbeat interval (offline shows within ~15s instead of 2 min)

    # ---- events / confirmation ------------------------------------------
    def _on_event(self, name: str, kind: str, data) -> None:
        if self.cfg.log_events:
            self._log_event(name, kind, data)
        if self.event_sink:
            self.event_sink(name, kind, data)

    def _log_event(self, name: str, kind: str, data) -> None:
        """Observability (opt-in [logging] log_events): append the agent event to a JSONL. Best-effort —
        a logging failure never breaks a run."""
        import json
        import time
        from pathlib import Path
        try:
            p = Path(self.workdir) / ".falamus" / "events.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "agent": name, "kind": kind, "data": data},
                                   ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def _confirm(self, tool, args, reason) -> bool:
        if self.confirm_fn:
            return self.confirm_fn(tool, args, reason)
        return False   # no confirmation UI → reject to be safe

    def _guard(self):
        pol = SafetyPolicy(
            workdir=self.workdir,
            confirm_command=self.cfg.confirm_command,
            confirm_write=self.cfg.confirm_write,
            allowed_paths=self.cfg.allowed_paths,
            dev_mode=self.dev_mode,
        )
        self._policy = pol          # keep a reference so /dev can toggle live without a rebuild
        return make_guard(pol, self._confirm)

    # ---- build / rebuild ------------------------------------------------
    def build(self, resume_sid: str | None = None) -> None:
        if self.runtime is not None:
            self.runtime.close_mcp()    # drop the old session's external MCP connections before rebuilding
        # select the active prompt SET for this session BEFORE any prompt is read (personas/rules): cloud
        # backends use the (minimal) cloud set, local uses the local set. Fixed for the session → the static
        # prefix stays constant across turns (KV cache / prompt cache still warm).
        import falamus.prompt as prompt
        prompt.set_active(self.cfg.prompt_cloud if providers.is_cloud(self.client.backend)
                          else self.cfg.prompt_local)
        self.error_log = ErrorLog.for_workdir(self.workdir)   # workdir may have changed via /cd
        self.client.error_log = self.error_log                # so client logs connection events here too
        kw = dict(
            on_event=self._on_event, guard=self._guard(),
            auto_compact=self.cfg.auto_compact, compact_threshold=self.cfg.compact_threshold,
            max_depth=self.cfg.max_depth,
            main_max_iters=self.cfg.max_iters_main, sub_max_iters=self.cfg.max_iters_sub,
            read_chunk_chars=self.cfg.read_chunk_chars,
            lang=self.cfg.lang, cancel_check=self.cancel_event.is_set,
            error_log=self.error_log,
        )
        if resume_sid:
            self.runtime = AgentRuntime.resume(self.client, resume_sid, self.workdir, **kw)
        else:
            self.runtime = AgentRuntime.start(self.client, self.workdir, **kw)
        self.runtime.force_plain_chat = self.force_plain_chat   # no shell → tool-less plain chat
        self.runtime.enable_shell = self.enable_shell   # opt-in persistent interactive shell sessions
        self.orch = self.runtime.build_orchestrator()

    # ---- run / cancel ---------------------------------------------------
    def run_message(self, text: str) -> str:
        self.cancel_event.clear()
        # heartbeat first: if the server is offline, fail fast instead of hanging
        if not self.client.ping():
            self.server_online = False
            self.error_log.log("server", "offline", "run_message aborted: server not reachable")
            return i18n.t("server_offline")
        self.server_online = True
        assert self.orch is not None        # set by __init__ → build()
        try:
            return self.orch.run(text)
        except Exception as e:  # noqa: BLE001
            import traceback
            self.error_log.log("main", "exception", f"run_message crashed: {e}", traceback.format_exc())
            return f"[error] {e}"
        finally:
            self.cancel_event.clear()
            # persistent shell sessions live for ONE turn only → close them all (no cross-turn leak)
            if self.runtime is not None:
                self.runtime.shell_mgr.close_all()
            self._refresh_ollama_nctx_once()

    def _refresh_ollama_nctx_once(self) -> None:
        """After the first chat (which loaded the ollama model), re-read the real context window from
        /api/ps — detect() saw it cold/empty. Updates info.n_ctx (gauge) + the live main CM (compaction
        budget). One-shot; no-op for non-ollama or once done."""
        if self._nctx_refreshed or self.client.backend != "ollama":
            return
        self._nctx_refreshed = True
        if self.client.refresh_ollama_nctx():
            cm = getattr(self.orch, "cm", None)
            if cm is not None and self.client.info is not None:
                cm.n_ctx = self.client.info.n_ctx     # budget is a property → updates with n_ctx

    def request_cancel(self) -> None:
        self.cancel_event.set()

    # ---- /exit: write a conversation summary into falamus.md (written directly, no safety prompt) ----
    def _save_progress(self, out: Callable[[str], None]) -> None:
        raw: list[dict] = self.orch.messages if self.orch else []
        msgs = [m for m in raw
                if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
                and m["content"].strip()]
        if not msgs:
            return
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)[:8000]
        out(i18n.t("saving_progress"))
        try:
            resp = self.client.chat(
                [{"role": "user", "content": _summary_prompt() + transcript}],
                max_tokens=700, enable_thinking=False,
            )
            summary = (resp.content or "").strip()
        except Exception as e:  # noqa: BLE001
            # don't hang/crash on /exit — but DO record it (this is the non-streaming path where a dead
            # connection used to freeze the TUI; now keepalive aborts it and the failure lands here).
            self.error_log.log("main", "save_progress_failed", f"progress summary call failed: {e}")
            out(i18n.t("progress_saved"))   # release the UI; the summary was just skipped
            return
        if summary:
            update_last_progress(self.workdir, summary)
            out(i18n.t("progress_saved"))

    # ---- status bar (left) ----------------------------------------------
    def status_left(self) -> str:
        # version / session / workdir only — model name, [DEV] and the live activity each get their OWN
        # status-bar field (rendered by the TUI), so a long model name can't crowd the whole bar.
        sid = self.runtime.session.sid if self.runtime else "-"
        wd = Path(self.workdir).name
        return f"v{__version__} │ {sid} │ {wd}"

    def open_shells(self) -> int:
        """Number of live persistent shell sessions (status bar shows a yellow CLI indicator when > 0)."""
        return self.runtime.shell_mgr.open_count() if self.runtime else 0

    def model_short(self) -> str:
        """Just the model's leaf name (the TUI shows this as a width-capped marquee)."""
        return (self.info.name or "?").split("/")[-1]

    def context_usage(self) -> tuple[int, int, bool] | None:
        """Return (used tokens, n_ctx, near auto-compaction); None if unavailable."""
        n_ctx = self.info.n_ctx or 0
        cm = getattr(self.orch, "cm", None)
        if not n_ctx or cm is None:
            return None
        used = cm._last_server_tokens or 0
        near = used >= 0.9 * cm.budget
        return used, n_ctx, near

    # ---- commands -------------------------------------------------------
    def command(self, line: str, out: Callable[[str], None]) -> bool:
        """Handle a / command; out(text) prints to the UI; returns True to quit."""
        assert self.orch is not None        # set by __init__ → build()
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            self._save_progress(out)
            return True
        elif cmd == "/help":
            out(i18n.t("help"))
        elif cmd == "/version":
            out(f"falamus {__version__}")
        elif cmd == "/tools":
            out(i18n.t("tools_label", tools=", ".join(self.orch.allowed_tools or [])))
        elif cmd == "/kill":
            mgr = self.runtime.shell_mgr if self.runtime else None
            shell_rows = mgr.listing() if mgr else []
            if not arg:
                if not shell_rows:
                    out(i18n.t("kill_none"))
                else:
                    out(i18n.t("kill_list"))
                    for pid, owner, note in shell_rows:
                        out(f"  {pid}  [{owner}] {note or '-'}")
                    out(i18n.t("kill_usage"))
            elif arg.isdigit() and mgr is not None:
                out(mgr.kill(int(arg)))
            else:
                out(i18n.t("kill_usage"))
        elif cmd == "/config":
            if not arg:
                out(self.cfg.to_text() + "\n\nset with: /config <key> <value>   (keys: "
                    + ", ".join(SETTABLE) + ")\n" + i18n.t("config_restart_note"))
            else:
                kv = arg.split(maxsplit=1)
                if len(kv) < 2:
                    out("usage: /config <key> <value>   (or /config alone to view all)")
                else:
                    key, val = kv[0], kv[1].strip()
                    try:
                        msg = apply_setting(self.cfg, key, val)
                        self.cfg.save()
                        if key == "lang":
                            i18n.set_lang(self.cfg.lang)
                        out(f"ok {msg}")
                        if key != "lang":
                            out(i18n.t("config_restart_note"))
                    except KeyError:
                        out(f"unknown setting '{key}'. valid: {', '.join(SETTABLE)}")
                    except ValueError as e:
                        out(f"bad value for {key}: {e}")
        elif cmd == "/compact":
            if self.orch.cm:
                out(i18n.t("compacting"))
                self.orch.messages = self.orch.cm.force(self.orch.messages)
                # push the new (smaller) token count to the status bar — /compact makes no model call, so
                # without this the bar keeps showing the pre-compact number (and looks like nothing happened)
                self._on_event("main", "usage", self.orch.cm.current_tokens(self.orch.messages))
                out(i18n.t("compacted"))
            else:
                out(i18n.t("no_compact"))
        elif cmd == "/reset":
            self.build()
            out(i18n.t("new_session"))
        elif cmd == "/sessions":
            rows = list_sessions(self.workdir)
            if not rows:
                out(i18n.t("no_sessions"))
            for s in rows:
                out(f"  {s['sid']}  {s['title'][:36]}")
        elif cmd == "/resume":
            if not arg:
                out(i18n.t("resume_usage"))
            else:
                self.build(resume_sid=arg)
                out(i18n.t("resumed", sid=arg))
                # the full history is restored into the model's context; replay only the TAIL to the screen
                # so you can see where you left off (without re-printing the whole conversation).
                tail = [m for m in (self.orch.messages if self.orch else [])
                        if m.get("role") in ("user", "assistant")
                        and isinstance(m.get("content"), str) and m["content"].strip()]
                if tail:
                    out(i18n.t("resume_tail"))
                    for m in tail[-4:]:
                        who = "you" if m["role"] == "user" else "main"
                        body = m["content"].strip()
                        out(f"{who}: {body[:1500]}")
        elif cmd == "/cd":
            p = Path(arg).expanduser()
            target = p if p.is_absolute() else Path.cwd() / p
            if arg and is_in_program(target):
                out(i18n.t("workdir_in_program"))
            elif arg and p.is_dir():
                self.workdir = str(p.resolve())
                self.cfg.workdir = self.workdir
                self.cfg.save()
                self.build()
                out(i18n.t("cd_done", wd=self.workdir))
            else:
                out(i18n.t("cd_bad", arg=arg))
        elif cmd == "/lang":
            if arg not in ("en", "zh"):
                out(i18n.t("lang_usage"))
            else:
                i18n.set_lang(arg)
                self.cfg.lang = arg
                self.cfg.save()
                out(i18n.t("lang_set", lang=arg))
        elif cmd == "/dev":
            self.dev_mode = not self.dev_mode
            if self._policy is not None:
                self._policy.dev_mode = self.dev_mode
            out(i18n.t("dev_on") if self.dev_mode else i18n.t("dev_off"))
        elif cmd == "/save":
            self.cfg.save()
            out(i18n.t("saved_config"))
        else:
            out(i18n.t("unknown_cmd", cmd=cmd))
        return False
