"""Safety mechanism.

Three lines of defense:
  1. Command blacklist: highly destructive commands are denied outright.
  2. Path allowlist: reads/writes outside the working directory require confirmation.
  3. Dangerous-action confirmation: run_command / write / edit require user confirmation by default (can be off).

`SafetyPolicy.evaluate()` returns a decision ("allow"|"confirm"|"deny", reason).
`make_guard()` wraps the policy + a confirm function into a registry guard(tool, args)->ToolResult|None.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from falamus.tools.registry import Tool, ToolResult

# highly destructive commands (denied outright) — the rules live in a USER-EDITABLE markdown file,
# blacklist.md: the packaged copy is the template, seeded once into ~/.config/falamus/ (seed_blacklist);
# the user copy — if present — is what's loaded. One regex per line; '#' comments; '#win:' = a rule
# applied only on Windows. This lets a user tighten (add rules) or loosen (remove rules) their own guardrail.
_BLACKLIST_FILE = "blacklist.md"


def _user_blacklist_path() -> Path:
    from falamus.settings import CONFIG_PATH
    return CONFIG_PATH.parent / _BLACKLIST_FILE


def _blacklist_text() -> str:
    user = _user_blacklist_path()
    if user.is_file():
        return user.read_text(encoding="utf-8")
    return (files(__package__) / _BLACKLIST_FILE).read_text(encoding="utf-8")


def _load_blacklist() -> list[str]:
    """Parse blacklist.md → regex list. ONLY lines inside a fenced ``` code block are rules (prose outside
    is documentation). Cross-platform lines + '#win:' lines (only on Windows). A line that won't compile is
    skipped, so a bad user edit can never crash or hang the guard."""
    pats: list[str] = []
    in_fence = False
    for raw in _blacklist_text().splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence or not line:
            continue
        if line.startswith("#win:"):
            if not _on_windows():
                continue
            line = line[len("#win:"):].strip()
        elif line.startswith("#"):
            continue
        try:
            re.compile(line)
        except re.error:
            continue
        pats.append(line)
    return pats


def seed_blacklist() -> None:
    """Copy the packaged blacklist.md into the user dir (~/.config/falamus/) once, so there's ONE editable
    place. Idempotent + best-effort (loading still falls back to the packaged copy). Called at startup."""
    dst = _user_blacklist_path()
    if dst.exists():
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((files(__package__) / _BLACKLIST_FILE).read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        pass
# destructive commands: confirm even in dev mode (rm, etc.) — not blocked, just need a nod
_DESTRUCTIVE = [
    r"\brm\b", r"\brmdir\b", r"\bmkfs\b", r"\bdd\b", r"\bshred\b",
    r"\btruncate\b", r"\bgit\s+reset\s+--hard\b",
    # mv only when a target is an ABSOLUTE path (relative moves inside the workdir don't nag in dev mode)
    r"\bmv\s.*\s/",
    # stdout redirect to a SENSITIVE system dir. The lookbehind excludes stderr/merge redirects (2> &>),
    # so harmless `2>/dev/null`, `>/dev/null`, `> /tmp/…` no longer trip it.
    r"(?<![0-9&])>>?\s*/(etc|boot|sys|usr|bin|sbin|lib|lib64|proc|root|var)\b",
]

# Windows-only DESTRUCTIVE patterns (the dev-mode "confirm" tier) — applied only on Windows. The Windows
# deny-blacklist rules now live in blacklist.md as `#win:` lines (see _load_blacklist).
_DESTRUCTIVE_WIN = [
    r"(?i)\b(?:del|erase)\b", r"(?i)\b(?:rd|rmdir)\b", r"(?i)\bformat\s+[a-z]:",
    r"(?i)\bRemove-Item\b", r"(?i)\bClear-Content\b",
    r"(?i)>\s*[a-z]:\\(?:Windows|Program)",                   # redirect into a Windows system dir (native)
    r"(?i)>\s*/[a-z]/(?:Windows|Program)",                    # ditto via a Git-Bash drive mount
]


def _on_windows() -> bool:
    """Whether to apply the Windows-only safety patterns (monkeypatch-friendly for tests)."""
    return os.name == "nt"
# tools whose path comes from a plain `path` ARG. write_file/append_file are NOT here: they carry the
# path inside `content` as a <<<FILE:path>>> header, so the policy parses it via _effective_path().
_PATH_ARGS = {
    "read_file": ["path"], "edit_file": ["path"],
    "list_dir": ["path"], "view_image": ["path"],
}
# file-writing tools that require write-confirmation and a path-allowlist check
_WRITE_TOOLS = ("write_file", "append_file", "edit_file")


@dataclass
class SafetyPolicy:
    workdir: str = "."
    confirm_command: bool = True
    confirm_write: bool = True
    allowed_paths: list[str] = field(default_factory=lambda: ["."])
    blacklist: list[str] = field(default_factory=_load_blacklist)
    dev_mode: bool = False          # developer mode: auto-allow everything except destructive commands

    def _abs_allowed(self) -> list[Path]:
        base = Path(self.workdir).expanduser().resolve()
        out = []
        for p in self.allowed_paths:
            pp = Path(p).expanduser()
            out.append((base / pp).resolve() if not pp.is_absolute() else pp.resolve())
        return out

    def _effective_path(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """The filesystem path a tool will actually touch. write_file/append_file carry it inside
        `content` as a <<<FILE:path>>> header (no `path` arg), so parse that; others use a path arg."""
        if tool_name in ("write_file", "append_file"):
            from falamus.tools.files import _unpack
            return _unpack(args)[0] or None
        for key in _PATH_ARGS.get(tool_name, []):
            v = args.get(key)
            if v:
                return str(v)
        return None

    def _in_allowed(self, path: str) -> bool:
        try:
            target = Path(path).expanduser()
            base = Path(self.workdir).expanduser().resolve()
            target = (base / target).resolve() if not target.is_absolute() else target.resolve()
        except (OSError, ValueError):
            return False
        for allowed in self._abs_allowed():
            if target == allowed or allowed in target.parents:
                return True
        return False

    def evaluate(self, tool: Tool, args: dict[str, Any]) -> tuple[str, str]:
        # 1) blacklist (shell only) → always deny
        if tool.name == "run_command":
            cmd = str(args.get("command", ""))
            for pat in self.blacklist:
                if re.search(pat, cmd):
                    return ("deny", f"blocked highly destructive command (matched rule /{pat}/)")

        # persistent interactive shell: confirm ONCE on open, with a heavy warning — afterwards the
        # session's input is NOT individually safety-checked, so this is the single gate. Confirmed even
        # in dev mode (placed before the dev-mode shortcut) because it opens an unchecked live terminal.
        if tool.name == "shell_open":
            cmd = str(args.get("command", ""))
            # the launch command itself still gets the destructive blacklist (so an auto-approving caller
            # can't open `rm -rf /` through it) — even though the session's LATER input cannot be checked.
            for pat in self.blacklist:
                if re.search(pat, cmd):
                    return ("deny", f"blocked highly destructive command (matched rule /{pat}/)")
            return ("confirm",
                    f"⚠ open a PERSISTENT interactive shell ({cmd}) — its later input is only BEST-EFFORT "
                    "checked (a blacklist, easily bypassed inside a live shell/REPL); the agent can largely "
                    "run what it wants inside it until it is closed at the end of this turn")

        # persistent-shell input: best-effort blacklist on each line sent (the session is otherwise the
        # single-gate model — see shell_open's warning). A hit needs the user's nod (confirm).
        if tool.name == "shell_input":
            text = str(args.get("input", "") or "")
            for pat in self.blacklist:
                if re.search(pat, text):
                    return ("confirm", f"⚠ destructive command sent to the live shell: {text.strip()}")
            return ("allow", "")

        # developer mode: destructive commands still need confirmation, everything else allowed
        if self.dev_mode:
            if tool.name == "run_command":
                cmd = str(args.get("command", ""))
                destructive = _DESTRUCTIVE + (_DESTRUCTIVE_WIN if _on_windows() else [])
                for pat in destructive:
                    if re.search(pat, cmd):
                        return ("confirm", f"(dev) destructive command needs confirmation: {cmd}")
            return ("allow", "")

        # 2) path allowlist (write-type outside workdir → confirm)
        path = self._effective_path(tool.name, args)
        if path and tool.risk in ("medium", "high") and not self._in_allowed(path):
            return ("confirm", f"writing / operating on a path outside the working directory: {path}")

        # 3) dangerous-action confirmation
        # external MCP tool (bridged, namespaced <server>__<tool> — no built-in has '__'): a new,
        # opaque attack surface → confirm every call. (dev_mode returned above, so /dev auto-allows.)
        if "__" in tool.name:
            return ("confirm", f"call external MCP tool: {tool.name}")
        if tool.name == "run_command" and self.confirm_command:
            return ("confirm", f"run command: {args.get('command', '')}")
        if tool.name in _WRITE_TOOLS and self.confirm_write:
            return ("confirm", f"{tool.name}: {self._effective_path(tool.name, args) or '?'}")

        return ("allow", "")


def make_guard(
    policy: SafetyPolicy,
    confirm_fn: Callable[[Tool, dict[str, Any], str], bool] | None,
) -> Callable[[Tool, dict[str, Any]], ToolResult | None]:
    """Wrap policy + confirm function into a registry guard.

    confirm_fn(tool, args, reason)->bool: True to allow, False to reject.
    If None (non-interactive), "confirm" is treated as auto-allow (whether confirmation is off is
    decided by config).
    Messages are actionable ("do NOT retry …") so a denied/declined call doesn't loop until max_iters.
    """
    def guard(tool: Tool, args: dict[str, Any]) -> ToolResult | None:
        action, reason = policy.evaluate(tool, args)
        if action == "deny":
            return ToolResult.error(
                f"[safety] blocked: {reason}. This is permanently blocked — do NOT retry; use a safer command.")
        if action == "confirm" and confirm_fn is not None:
            if not confirm_fn(tool, args, reason):
                return ToolResult.error(
                    "[safety] the user declined this action. Do NOT retry the same thing — try a different "
                    "approach or ask the user what they want.")
        return None
    return guard


if __name__ == "__main__":
    from falamus.tools import default_registry

    reg = default_registry(".")
    pol = SafetyPolicy(workdir=".", confirm_command=True, confirm_write=True)

    def fake_confirm(tool, args, reason):
        print(f"  ❓ confirm: {reason} → (auto-reject)")
        return False

    reg.guard = make_guard(pol, fake_confirm)

    class _C:
        def __init__(self, name, **a):
            self.name, self.arguments, self.id = name, a, "x"

    print("blacklist block:")
    print(" ", reg.execute(_C("run_command", command="rm -rf /")).text)
    print("needs confirmation (rejected):")
    print(" ", reg.execute(_C("run_command", command="echo hi")).text)
    print("low-risk allowed (list_dir):")
    print(" ", reg.execute(_C("list_dir", path=".")).text.splitlines()[0])
    print("write outside workdir needs confirmation:")
    print(" ", reg.execute(_C("write_file", path="/etc/xx", content="y")).text)
