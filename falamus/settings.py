"""Config loading (this is code — do not edit it as a config file).

The config file lives in the USER's config dir (``$XDG_CONFIG_HOME/falamus/config.ini`` or
``~/.config/falamus/config.ini``) — NOT inside the installed package, so it survives reinstalls
and is the same regardless of where the package is installed. Users normally don't edit it by hand:
every setting is reachable via a ``falamus --<flag>`` CLI flag or the TUI ``/config`` command
(see SETTABLE / apply_setting below). Created with defaults on first run.

The working directory (workdir) is set via the positional arg / ``--workdir`` / TUI and saved here.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # the installed package dir falamus/
PROGRAM_ROOT = PKG_DIR.parent                       # outer level (same dir as run.py), for is_in_program


def _is_windows() -> bool:
    """OS check, isolated so tests can patch it without monkeypatching os.name (which would make pathlib
    try to build a WindowsPath on a POSIX host)."""
    return os.name == "nt"


def _user_config_path() -> Path:
    # Honour XDG_CONFIG_HOME on any platform; otherwise use the OS-native config dir:
    # Windows → %APPDATA%\falamus, POSIX → ~/.config/falamus.
    xdg = os.environ.get("XDG_CONFIG_HOME")
    appdata = os.environ.get("APPDATA")
    if xdg:
        base = Path(xdg)
    elif _is_windows() and appdata:
        base = Path(appdata)
    else:
        base = Path("~/.config").expanduser()
    return base.expanduser() / "falamus" / "config.ini"


CONFIG_PATH = _user_config_path()                   # user config (not inside the package)


@dataclass
class Config:
    base_url: str = "http://localhost:8080"
    backend: str = "auto"            # auto|llama_cpp|ollama|anthropic — declared in core so the schema is
    model: str = ""                  # stable (installing the [cloud] extra never rewrites/resets the INI)
    max_tokens: int = -1             # internal fallback only (not a user setting); -1 = unbounded, matching agents
    thinking: str = "off"            # off|low|medium|high (this model only supports on/off; level reserved)
    auto_compact: bool = True
    compact_threshold: float = 0.7
    confirm_command: bool = True
    confirm_write: bool = True
    allowed_paths: list[str] = field(default_factory=lambda: ["."])
    max_depth: int = 1               # default: main spawns leaf sub-agents only (no nesting) — lightweight. Raise for deeper chains.
    max_iters_main: int = 0          # main-agent tool-iteration cap; 0 = UNLIMITED (its guard is the circuit breaker, not a cap)
    max_iters_sub: int = 60          # sub-agent tool-iteration cap (backstop against single-unit loops)
    read_chunk_chars: int = 8000     # suggested read-chunk size for read_file (offset/limit chunked reads)
    persistent_interactive_shell: bool = False       # persistent interactive shell tools (shell_open/input/close); default off; POSIX only
    log_events: bool = False         # observability: append every agent event to <workdir>/.falamus/events.jsonl
    prompt_local: str = "default_local"   # which prompt set to use for local backends (--set-local-prompt)
    prompt_cloud: str = "default_cloud"   # which prompt set to use for cloud backends (--set-cloud-prompt)
    repeat_penalty: float = 1.15     # sampler: >1 discourages repeating recent tokens (1.0 = off)
    repeat_last_n: int = 256         # sampler: how many recent tokens the repeat penalty looks back over
    workdir: str = ""                # empty = not set yet, asked on startup
    lang: str = "en"                 # en | zh

    # ---- load / save (INI) ---------------------------------------------
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> Config:
        """Read the INI; if absent, create a default file and return the defaults."""
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        cp = configparser.ConfigParser(inline_comment_prefixes=(";",))  # allow `key = value  ; comment`
        cp.read(path, encoding="utf-8")

        def g(sec: str, key: str, default: str) -> str:
            return cp.get(sec, key, fallback=default)

        def gb(sec: str, key: str, default: bool) -> bool:
            return cp.getboolean(sec, key, fallback=default)

        d = cls()
        return cls(
            base_url=g("backend", "base_url", d.base_url),
            backend=g("backend", "backend", d.backend),
            model=g("backend", "model", d.model),
            max_tokens=cp.getint("backend", "max_tokens", fallback=d.max_tokens),
            thinking=g("backend", "thinking", d.thinking),
            auto_compact=gb("context", "auto_compact", d.auto_compact),
            compact_threshold=cp.getfloat("context", "compact_threshold", fallback=d.compact_threshold),
            confirm_command=gb("safety", "confirm_command", d.confirm_command),
            confirm_write=gb("safety", "confirm_write", d.confirm_write),
            allowed_paths=[p.strip() for p in g("safety", "allowed_paths", ".").split(",") if p.strip()],
            max_depth=cp.getint("agents", "max_depth", fallback=d.max_depth),
            max_iters_main=cp.getint("agents", "max_iters_main", fallback=d.max_iters_main),
            max_iters_sub=cp.getint("agents", "max_iters_sub", fallback=d.max_iters_sub),
            read_chunk_chars=cp.getint("tools", "read_chunk_chars", fallback=d.read_chunk_chars),
            persistent_interactive_shell=gb("tools", "persistent_interactive_shell", d.persistent_interactive_shell),
            prompt_local=g("prompt", "local", d.prompt_local),
            prompt_cloud=g("prompt", "cloud", d.prompt_cloud),
            repeat_penalty=cp.getfloat("sampling", "repeat_penalty", fallback=d.repeat_penalty),
            repeat_last_n=cp.getint("sampling", "repeat_last_n", fallback=d.repeat_last_n),
            workdir=g("workspace", "workdir", d.workdir),
            lang=g("ui", "lang", d.lang),
            log_events=gb("logging", "log_events", d.log_events),
        )

    def save(self, path: Path = CONFIG_PATH) -> None:
        # Start from the EXISTING file so keys/sections we don't manage (e.g. added by a future extra, or
        # hand-added) survive a save instead of being wiped — the INI schema is stable across extra installs.
        cp = configparser.ConfigParser()
        if path.exists():
            try:
                cp.read(path, encoding="utf-8")
            except configparser.Error:
                cp = configparser.ConfigParser()

        def put(section: str, values: dict[str, str]) -> None:
            if not cp.has_section(section):
                cp.add_section(section)
            for k, v in values.items():
                cp.set(section, k, v)

        # backend/model are declared in core (stable schema) even without the [cloud] extra. max_tokens is
        # deliberately NOT written/exposed: every agent call carries its own cap (-1), so it never fires.
        put("backend", {"base_url": self.base_url, "backend": self.backend,
                        "model": self.model, "thinking": self.thinking})
        put("context", {"auto_compact": str(self.auto_compact).lower(),
                        "compact_threshold": str(self.compact_threshold)})
        put("safety", {"confirm_command": str(self.confirm_command).lower(),
                       "confirm_write": str(self.confirm_write).lower(),
                       "allowed_paths": ", ".join(self.allowed_paths)})
        put("agents", {"max_depth": str(self.max_depth),
                       "max_iters_main": str(self.max_iters_main),
                       "max_iters_sub": str(self.max_iters_sub)})
        put("tools", {"read_chunk_chars": str(self.read_chunk_chars),
                      "persistent_interactive_shell": str(self.persistent_interactive_shell).lower()})
        put("prompt", {"local": self.prompt_local, "cloud": self.prompt_cloud})
        put("sampling", {"repeat_penalty": str(self.repeat_penalty),
                         "repeat_last_n": str(self.repeat_last_n)})
        put("workspace", {"workdir": self.workdir})
        put("ui", {"lang": self.lang})
        put("logging", {"log_events": str(self.log_events).lower()})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("; Local LLM Helper config (edit freely)\n")
            f.write("; workdir empty => asked on startup; auto-saved once set\n")
            f.write(";\n")
            f.write("; --- anti-repetition (stops a weak model looping forever) ---\n")
            f.write("; [sampling] PREVENTS repetition while generating (sent to the model server):\n")
            f.write(";   repeat_penalty  : >1 penalises repeating recent tokens; 1.0 = off. Try 1.1-1.3.\n")
            f.write(";   repeat_last_n   : how many recent tokens that penalty looks back over (bigger = catches\n")
            f.write(";                     longer-range loops; the server default 64 is too short for paragraphs).\n")
            f.write("; (The runaway-loop DETECTOR that aborts a spiral is now automatic — no settings.)\n\n")
            cp.write(f)

    def to_text(self) -> str:
        return CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else "(no config file)"


# ── settable fields: field → (CLI flag, kind). Drives `falamus --<flag>` and TUI `/config <field>`. ──
# Single source so every INI setting is reachable from CLI and TUI (no hand-editing the file needed).
# field → (CLI flag, kind, what it does, example value)
SETTABLE: dict[str, tuple[str, str, str, str]] = {
    # NOTE: base_url / backend / model are set via the `falamus --set-provider` wizard (one unified entry),
    # NOT as individual flags — so they're deliberately absent from SETTABLE (keeps --help uncluttered).
    "thinking":         ("--thinking", "str", "reasoning effort (this model: on/off)", "off|low|medium|high"),
    "auto_compact":     ("--auto-compact", "bool", "auto-compact context near the limit", "true"),
    "compact_threshold":("--compact-threshold", "float", "compact when used ≥ this fraction of ctx", "0.7"),
    "confirm_command":    ("--confirm-command", "bool", "ask before running a command", "true"),
    "confirm_write":    ("--confirm-write", "bool", "ask before writing files", "true"),
    "allowed_paths":    ("--allowed-paths", "list", "dirs the agent may write in without confirming "
                         "(comma-separated; . = the workdir)", ".,/tmp"),
    "max_depth":        ("--max-depth", "int", "max sub-agent chain depth (0 = no sub-agents)", "1"),
    "max_iters_main":   ("--max-iters-main", "int", "main tool-iteration cap (0=unlimited)", "0"),
    "max_iters_sub":    ("--max-iters-sub", "int", "sub-agent tool-iteration cap", "60"),
    "read_chunk_chars": ("--read-chunk-chars", "int", "suggested read_file chunk size", "8000"),
    "persistent_interactive_shell":     ("--persistent-interactive-shell", "bool", "enable persistent interactive shell tools (POSIX)", "false"),
    "log_events":       ("--log-events", "bool", "log agent events to <workdir>/.falamus/events.jsonl", "false"),
    "repeat_penalty":   ("--repeat-penalty", "float", "sampler: >1 discourages repetition (1.0=off)", "1.15"),
    "repeat_last_n":    ("--repeat-last-n", "int", "sampler: tokens the repeat penalty looks back over", "256"),
    "workdir":          ("--workdir", "str", "default working directory", "/path/to/project"),
    "lang":             ("--lang", "str", "UI language", "en|zh"),
}
_BOOL_TRUE = {"1", "true", "yes", "on", "y"}
_BOOL_FALSE = {"0", "false", "no", "off", "n"}


def _coerce(field_name: str, raw: str):
    kind = SETTABLE[field_name][1]
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        v = raw.strip().lower()
        if v in _BOOL_TRUE:
            return True
        if v in _BOOL_FALSE:
            return False
        raise ValueError(f"expected true/false (got {raw!r})")
    if kind == "list":
        return [p.strip() for p in raw.split(",") if p.strip()]
    return raw  # str


def apply_setting(cfg: Config, field_name: str, raw: str) -> str:
    """Set ONE config field from a raw string (shared by CLI and the TUI /config command).

    Returns a 'field = value' confirmation. Raises KeyError (unknown field) or ValueError (bad value).
    """
    if field_name not in SETTABLE:
        raise KeyError(field_name)
    setattr(cfg, field_name, _coerce(field_name, raw))
    return f"{field_name} = {getattr(cfg, field_name)}"


def flag_to_field() -> dict[str, str]:
    """Reverse map `--flag` → field name, for CLI parsing."""
    return {spec[0]: f for f, spec in SETTABLE.items()}


if __name__ == "__main__":
    import tempfile
    p = Path(tempfile.mkdtemp()) / "config.ini"
    c = Config(base_url="http://localhost:8080", lang="zh")
    c.save(p)
    print(p.read_text())
    loaded = Config.load(p)
    print("loaded back:", loaded.base_url, "workdir=", repr(loaded.workdir), "lang=", loaded.lang)
    print("PROGRAM_ROOT:", PROGRAM_ROOT)
