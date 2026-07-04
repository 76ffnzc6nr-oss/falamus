"""CLI entry point.

Interactive terminal → full-screen TUI (ui/tui.py): UI decoupled from agents, ESC interrupt,
combined multi-agent display, waiting animation.
Non-terminal (pipe/redirect) → falls back to plain-text line mode.

Config comes from the user config dir (not a project file; see below) — there's no interactive
setup wizard. If the workdir is unset / nonexistent, you're asked on startup; saved back once set.
Default language is English (switch with /lang).

Config lives at ~/.config/falamus/config.ini (or $XDG_CONFIG_HOME) — set it via --<flag> or the TUI
/config command, not by hand.

Usage: falamus [workdir] [--resume <sid>]
"""

from __future__ import annotations

import atexit
import sys
from pathlib import Path

import falamus.i18n as i18n
from falamus.core import providers, secrets
from falamus.core.backend import Backend, is_in_program
from falamus.core.client import BackendError, LLMClient
from falamus.settings import CONFIG_PATH, SETTABLE, Config, apply_setting, flag_to_field


def _pick_model(loaded: list[str], tags: list[str], choose) -> str | None:
    """Decide which ollama model to pin. If one is already LOADED, use it (don't force a switch).
    Otherwise pick from the pulled models: the only one, or — when there are several — `choose(tags)`.
    Returns None when there's nothing to decide (let auto-detect handle it)."""
    if loaded:
        return loaded[0]
    if not tags:
        return None
    if len(tags) == 1:
        return tags[0]
    return choose(tags)


def _interactive_tty(interactive: bool) -> bool:
    return interactive and sys.stdin.isatty() and sys.stdout.isatty()


def _choose_model(items: list[str], interactive: bool) -> str | None:
    """Pick a model at startup: an interactive numbered menu when there's a real choice (>1, on a tty),
    otherwise the first. A non-tty never prompts."""
    if not items:
        return None
    if len(items) == 1 or not _interactive_tty(interactive):
        return items[0]
    return items[_menu(i18n.t("ollama_pick_model"), items)]


def _select_ollama_model(base_url: str, interactive: bool) -> str | None:
    """For ollama at STARTUP: use the already-loaded model if any (no prompt), else pick from the pulled
    models (interactive menu, like the cloud picker), then PRELOAD if cold. None → non-ollama/unreachable."""
    probe = LLMClient(base_url)
    try:
        if probe._detect_backend() != "ollama":
            return None
    except Exception:
        return None   # connection error → let Backend.detect surface it cleanly

    loaded = probe.loaded_models()
    model = loaded[0] if loaded else _choose_model(probe.available_models(), interactive)
    if model and model not in loaded:
        # cold model → load it now (blocks ~load time) with a visible message, instead of entering with a
        # wrong ctx (cold /api/ps is empty → n_ctx falls back to the arch max) and a slow first turn.
        print(i18n.t("loading_model", model=model))
        probe.preload_model(model)
    return model


def _select_cloud_model(cfg: Config, interactive: bool) -> str | None:
    """For a cloud backend at STARTUP: query the models this key can use and pick one (like ollama), falling
    back to the provider default. Missing key → None (Backend surfaces the actionable error)."""
    prov = providers.get(cfg.backend)
    if prov is None:
        return None
    try:
        key = secrets.load_api_key(cfg.backend)
    except Exception:
        key = None
    if not key:
        return None
    try:
        models = LLMClient(prov.endpoint, backend=cfg.backend, api_key=key).list_models()
    except Exception:
        models = []
    return _choose_model(models, interactive) or prov.default_model


def _connect(cfg: Config, force_plain_chat: bool = False) -> Backend:
    """Build the Backend (which probes the server); on an unreachable/undetectable server, print a clean
    one-liner with the fix instead of a traceback, and exit."""
    # model is chosen at STARTUP (interactive menu, like ollama): cloud → GET /v1/models; local → ollama.
    # interactive only when there's a real terminal (no TTY → pick the first, e.g. the text fallback).
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    model = (_select_cloud_model(cfg, interactive=interactive) if providers.is_cloud(cfg.backend)
             else _select_ollama_model(cfg.base_url, interactive=interactive))
    try:
        return Backend(cfg, force_plain_chat=force_plain_chat, model=model)
    except BackendError as e:
        # cloud key/extra errors carry an actionable message; local unreachable uses the generic hint
        msg = str(e) if providers.is_cloud(cfg.backend) else i18n.t("server_unreachable", url=cfg.base_url)
        print("falamus: " + msg, file=sys.stderr)
        sys.exit(1)


def _set_prompt(cfg: Config) -> None:
    """Pick which prompt set the local / cloud backends use (merged; a step inside --config). Scans built-in
    + ~/.config/falamus/prompts/ sets. Saved to config."""
    import falamus.prompt as prompt
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("falamus: prompt-set selection needs an interactive terminal.", file=sys.stderr)
        sys.exit(2)
    udir = prompt.user_prompts_dir()
    print(f"\nPrompt sets live in: {udir}")
    print("  (to make your own: copy one of the folders there, rename it, and edit its .md files —")
    print("   rules.md / agents.md / tools.md / notes.md — then pick it below.)")
    which = "local" if _menu("Prompt set for which models?", ["Local models", "Cloud models"]) == 0 else "cloud"
    sets = prompt.list_sets()
    cur = cfg.prompt_local if which == "local" else cfg.prompt_cloud
    chosen = sets[_menu(f"Prompt set for {which} models:",
                        [f"{s}  (current)" if s == cur else s for s in sets])]
    if which == "local":
        cfg.prompt_local = chosen
    else:
        cfg.prompt_cloud = chosen
    cfg.save()
    print(f"falamus: {which} prompt set = {chosen}", file=sys.stderr)


def _config_wizard(cfg: Config) -> None:
    """`falamus --config` — interactive settings menu (so --help needn't list every flag). Loops: pick a
    setting → enter a value, or pick 'prompt set'. Each change is saved. Plain `--<setting> <value>` still
    works for scripting. Blank input = done."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("falamus --config needs an interactive terminal.", file=sys.stderr)
        sys.exit(2)
    fields = sorted(SETTABLE)   # A-Z
    while True:
        print("\nConfigure falamus:")
        for i, f in enumerate(fields):
            print(f"  [{i}] {f} = {getattr(cfg, f)}")
        print(f"  [{len(fields)}] prompt set (local/cloud)")
        print("  (blank = done)")
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not raw:
            return
        if not raw.isdigit() or int(raw) > len(fields):
            continue
        idx = int(raw)
        if idx == len(fields):
            _set_prompt(cfg)
            continue
        field = fields[idx]
        _flag, _kind, desc, example = SETTABLE[field]
        try:
            val = input(f"{field} — {desc}\n  current: {getattr(cfg, field)}   (e.g. {example})\n"
                        "  new value (blank = keep): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not val:
            continue
        try:
            msg = apply_setting(cfg, field, val)
        except (KeyError, ValueError) as e:
            print(f"falamus: bad value: {e}", file=sys.stderr)
            continue
        cfg.save()
        if field == "lang":
            i18n.set_lang(cfg.lang)
        print(f"falamus: {msg} (saved)")


def _menu(title: str, options: list[str]) -> int:
    """Print a numbered menu, return the chosen index (0 on empty/invalid/EOF)."""
    print(title)
    for i, o in enumerate(options):
        print(f"  [{i}] {o}")
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0
    return int(raw) if raw.isdigit() and int(raw) < len(options) else 0


def _setup_provider(cfg: Config) -> None:
    """`falamus --set-provider` — the ONE setup entry. A small wizard: local vs cloud → connection details
    (local: url:port; cloud: pick provider + api-key) → saved to config.ini (+ key to the encrypted store).
    Afterwards `falamus` launches exactly what was configured; there is no auto-detect/pick at startup."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("falamus --set-provider needs an interactive terminal.", file=sys.stderr)
        sys.exit(2)
    if _menu("Set up falamus — where does the model run?",
             ["Local model (llama.cpp / ollama)", "Cloud (hosted API)"]) == 0:
        default = cfg.base_url or "http://localhost:8080"
        try:
            url = input(f"Local model server URL [{default}]: ").strip() or default
        except (EOFError, KeyboardInterrupt):
            url = default
        cfg.base_url = url
        try:                                   # record the concrete backend now so startup needn't probe
            cfg.backend = LLMClient(url)._detect_backend()
        except Exception:
            cfg.backend = "auto"
        cfg.model = ""                         # local model comes from the server
        cfg.save()
        print(f"falamus: local backend '{cfg.backend}' at {url} saved. Run: falamus", file=sys.stderr)
        return
    # cloud
    provs = [p for p in (providers.get(pid) for pid in providers.ids()) if p is not None]
    idx = _menu("Choose a cloud provider:", [p.display for p in provs] + ["(more coming soon)"])
    if idx >= len(provs):
        print("falamus: only Anthropic is available right now.", file=sys.stderr)
        sys.exit(2)
    prov = provs[idx]
    provider = prov.id
    import getpass
    try:
        key = getpass.getpass(f"{prov.display} API key (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""
    if not key:
        print("falamus: no key entered — nothing changed.", file=sys.stderr)
        sys.exit(2)
    try:
        secrets.save_api_key(provider, key)    # encrypted, write-only, one per provider (overwrites)
    except secrets.SecretsUnavailable as e:
        print(f"falamus: {e}", file=sys.stderr)
        sys.exit(1)
    cfg.backend = provider
    cfg.model = ""                             # model is chosen at startup (like ollama), not here
    cfg.save()
    print(f"falamus: cloud backend '{provider}' saved (key stored). Run: falamus — you'll pick the model "
          "at startup.", file=sys.stderr)

def _add_mcp(args: list[str]) -> None:
    """`falamus --add-mcp <name> -- <command> [args...]` — register an external MCP server (client direction)."""
    from falamus.mcp_config import add_mcp_server
    i = args.index("--add-mcp")
    name = args[i + 1] if i + 1 < len(args) else ""
    cmd = args[args.index("--") + 1:] if "--" in args else []
    if not name or name == "--" or not cmd:
        print("usage: falamus --add-mcp <name> -- <command> [args...]", file=sys.stderr)
        sys.exit(2)
    add_mcp_server(name, cmd[0], cmd[1:])
    print(f"falamus: added MCP server '{name}': {' '.join(cmd)}", file=sys.stderr)


def _list_mcp() -> None:
    from falamus.mcp_config import load_mcp_servers, mcp_servers_path
    servers = load_mcp_servers()
    if not servers:
        print(f"falamus: no MCP servers configured ({mcp_servers_path()})", file=sys.stderr)
        return
    for name, spec in servers.items():
        print(f"  {name}: {spec.get('command', '')} {' '.join(spec.get('args', []))}")


def _remove_mcp(args: list[str]) -> None:
    from falamus.mcp_config import remove_mcp_server
    i = args.index("--remove-mcp")
    name = args[i + 1] if i + 1 < len(args) else ""
    if not name:
        print("usage: falamus --remove-mcp <name>", file=sys.stderr)
        sys.exit(2)
    print(f"falamus: {'removed' if remove_mcp_server(name) else 'no such'} MCP server '{name}'", file=sys.stderr)


def _copy_prompt(args: list[str]) -> None:
    """`falamus --copy-prompt <builtin> <new-name>` — copy a built-in set to an editable user set (built-ins
    are read-only / read straight from the package)."""
    import falamus.prompt as prompt
    i = args.index("--copy-prompt")
    builtin = args[i + 1] if i + 1 < len(args) else ""
    new_name = args[i + 2] if i + 2 < len(args) else ""
    if not builtin or not new_name or new_name.startswith("-"):
        print("usage: falamus --copy-prompt <default_local|default_cloud> <new-name>", file=sys.stderr)
        sys.exit(2)
    try:
        dst = prompt.copy_builtin(builtin, new_name)
    except ValueError as e:
        print(f"falamus: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"falamus: copied {builtin} → {dst}\n  edit it, then select with --set-prompt", file=sys.stderr)


_HISTORY_FILE = Path("~/.falamus/history").expanduser()

_USAGE = """falamus {v}

Usage:
  falamus [workdir]                    start the interactive TUI in <workdir>
  falamus --resume <sid> [workdir]     resume a previous session
  falamus --set-provider               set up the model source (local url / cloud provider + key)
  falamus --config                     configure settings + prompt sets (interactive menu)
  falamus --mcp                        run as an MCP server over stdio (for another agent to drive)
  falamus --add-mcp <name> -- <cmd>    add an external MCP server (also --list-mcp / --remove-mcp <name>)
  falamus --<setting> <value> ...      set one config value directly (saved); e.g. --thinking on
  falamus --version | -V               print the version
  falamus --help | -h                  show this help

Notes:
  - the destructive-command blacklist is yours to edit (add/remove rules): {blacklist}"""


def _setup_history() -> None:
    try:
        import readline
    except ImportError:
        return
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(_HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)
    atexit.register(lambda: _safe_write_history(readline))


def _safe_write_history(readline) -> None:
    try:
        readline.write_history_file(_HISTORY_FILE)
    except OSError:
        pass


def ensure_workdir(cfg: Config) -> str:
    """Ensure a usable workdir; ask to create if missing, ask for a path if unset. Saved back to the INI."""
    while True:
        wd = cfg.workdir.strip()
        if not wd:
            wd = input(i18n.t("ask_workdir")).strip()
            if not wd:
                print(i18n.t("workdir_empty"))
                continue
        path = Path(wd).expanduser()
        target = path if path.is_absolute() else (Path.cwd() / path)
        if is_in_program(target):
            print(i18n.t("workdir_in_program"))
            cfg.workdir = ""
            continue
        path = target.resolve()
        if not path.exists():
            ans = input(i18n.t("workdir_missing", wd=path) + "\n" + i18n.t("ask_create")).strip().lower()
            if ans in ("y", "yes"):
                path.mkdir(parents=True, exist_ok=True)
                print(i18n.t("created_workdir", wd=path))
            else:
                cfg.workdir = ""
                continue
        cfg.workdir = str(path)
        cfg.save()
        return cfg.workdir


def _run_text_fallback(backend: Backend) -> None:
    """Non-terminal: plain-text line mode (no TUI / no background thread)."""
    backend.confirm_fn = lambda tool, args, reason: \
        input("\n" + i18n.t("confirm", reason=reason)).strip().lower() in ("y", "yes")
    backend.event_sink = None
    print("\n" + i18n.t("welcome") + "\n")
    while True:
        try:
            line = input(i18n.t("prompt")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + i18n.t("bye"))
            return
        if not line:
            continue
        if line.startswith("/"):
            if backend.command(line, print):
                print(i18n.t("bye"))
                return
            continue
        out = backend.run_message(line)
        print("\n" + i18n.t("agent_reply", out=out) + "\n")


def _enable_windows_ansi() -> None:
    """On Windows, let raw ANSI escapes (the plain-line status bar) render on legacy consoles. No-op on
    POSIX and on modern Windows Terminal; silently skips if colorama isn't installed."""
    import os
    if os.name != "nt":
        return
    try:
        import colorama
        colorama.just_fix_windows_console()
    except Exception:
        pass


def _check_shell() -> bool:
    """Startup self-check: on Windows, confirm a working bash is available for run_command. If not, warn that
    the built-in tools will be DISABLED and falamus will run as plain chat (the no-tools mode), and — when
    interactive — ask for a y/N confirmation. Returns True when tools should be disabled (plain-chat mode)."""
    from falamus.tools.cli import shell_healthcheck
    ok, detail = shell_healthcheck()
    if ok:
        return False
    print(f"\n⚠  {detail}", file=sys.stderr)
    print("   → No usable shell: the built-in tools are DISABLED; falamus opens in PLAIN CHAT (no tools).",
          file=sys.stderr)
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            ans = input("   Continue in plain-chat mode? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans != "y":
            print("Aborted.", file=sys.stderr)
            sys.exit(1)
    return True


def _check_persistent_shell(cfg: Config) -> None:
    """Warn at startup if the persistent interactive shell was enabled but can't run here (POSIX only) —
    otherwise enabling it on Windows is a silent no-op that just confuses the user."""
    from falamus.tools.shell_session import available
    if not cfg.persistent_interactive_shell or available():
        return
    print("\033[91m⚠  persistent_interactive_shell is on, but it is POSIX-only and is "
          "DISABLED on this platform.\033[0m", file=sys.stderr)
    print("   → You can get the same capability from an external MCP server that provides an interactive "
          "shell, and add it with --add-mcp.", file=sys.stderr)


def main() -> None:
    _enable_windows_ansi()
    args = list(sys.argv[1:])
    if "--version" in args or "-V" in args:
        from falamus.version import __version__
        print(f"falamus {__version__}")
        return
    from falamus.core import safety as _safety
    _safety.seed_blacklist()   # ensure the editable destructive-command blacklist exists in the user dir
    if "--help" in args or "-h" in args:
        from falamus.core.safety import _user_blacklist_path
        from falamus.version import __version__
        print(_USAGE.format(v=__version__, blacklist=_user_blacklist_path()))
        return
    if "--set-provider" in args:            # the one setup wizard (local/cloud) → saves config, then exit
        _cfg = Config.load()
        i18n.set_lang(_cfg.lang)
        _setup_provider(_cfg)
        return
    if "--config" in args:                  # interactive settings menu (settings + prompt sets)
        _cfg = Config.load()
        i18n.set_lang(_cfg.lang)
        _config_wizard(_cfg)
        return
    if "--set-prompt" in args:              # merged prompt-set picker (also reachable via --config)
        _set_prompt(Config.load())
        return
    if "--mcp" in args:                     # serve falamus as an MCP server over stdio (client spawns us)
        from falamus.mcp_server import main as mcp_main
        mcp_main()
        return
    if "--add-mcp" in args:                 # register / list / remove EXTERNAL MCP servers (client direction)
        _add_mcp(args)
        return
    if "--list-mcp" in args:
        _list_mcp()
        return
    if "--remove-mcp" in args:
        _remove_mcp(args)
        return
    if "--copy-prompt" in args:             # copy a built-in prompt set to an editable user set
        _copy_prompt(args)
        return
    resume_sid = None
    if "--resume" in args:
        i = args.index("--resume")
        resume_sid = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]
    # config-setting flags (table-driven, one per INI field) — extracted before the positional workdir
    f2f = flag_to_field()
    config_sets: dict[str, str] = {}
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in f2f:
            if i + 1 >= len(args):
                print(f"falamus: {args[i]} needs a value", file=sys.stderr)
                sys.exit(2)
            config_sets[f2f[args[i]]] = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    args = rest
    workdir_arg = args[0] if args else None

    cfg = Config.load()
    for fieldname, raw in config_sets.items():
        try:
            apply_setting(cfg, fieldname, raw)
        except (KeyError, ValueError) as e:
            print(f"falamus: bad value for {fieldname}: {e}", file=sys.stderr)
            sys.exit(2)
    if config_sets:
        cfg.save()
        print(f"falamus: saved [{', '.join(config_sets)}] → {CONFIG_PATH}", file=sys.stderr)
        if not workdir_arg:   # pure configure: nothing else to do
            return
    i18n.set_lang(cfg.lang)
    _check_persistent_shell(cfg)   # warn if the persistent shell is enabled but unavailable (POSIX only)
    plain_chat = _check_shell()   # no working shell → run tool-less (plain chat)
    if workdir_arg:
        cfg.workdir = workdir_arg

    ensure_workdir(cfg)

    print(i18n.t("connecting", url=cfg.base_url))
    backend = _connect(cfg, force_plain_chat=plain_chat)
    print("  " + backend.info.summary())
    if resume_sid:
        backend.build(resume_sid=resume_sid)
    assert backend.runtime is not None      # set by Backend.__init__ → build()
    print(i18n.t("session_line", sid=backend.runtime.session.sid, wd=backend.workdir))

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        from falamus.ui.tui import HelperTUI
        HelperTUI(backend).run()
    else:
        _setup_history()
        _run_text_fallback(backend)


if __name__ == "__main__":
    main()
