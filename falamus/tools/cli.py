"""CLI tool: run_command.

Runs a shell command in the working directory and returns stdout/stderr/exit code.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from falamus.prompt import frag

from .registry import Tool, ToolResult

_DEFAULT_TIMEOUT = 60
_MAX_OUTPUT = 30_000  # output cap (chars); beyond this keep head + tail


def _is_windows() -> bool:
    """Isolated OS check (monkeypatch-friendly for tests)."""
    return os.name == "nt"


def _prefer_bash(cands: list[str]) -> str | None:
    """From candidate bash paths, prefer a real Git-Bash/MSYS bash over the System32 WSL launcher (which
    is broken when no WSL distro is installed). Pure preference logic, isolated for testing."""
    non_wsl = [c for c in cands if "system32" not in c.lower()]
    chosen = non_wsl or cands
    return chosen[0] if chosen else None


def _find_bash() -> str | None:
    """Locate a POSIX bash on Windows, PREFERRING a real Git-Bash over the ``C:\\Windows\\System32\\bash.exe``
    WSL launcher — that launcher prints a "no installed distribution" message and exits 1 when no WSL distro
    is installed, which would make every run_command fail/hang. None if no bash is found."""
    cands: list[str] = []
    for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
        if os.path.isfile(p):
            cands.append(p)
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "bash.exe")
        if d and os.path.isfile(p):
            cands.append(p)
    return _prefer_bash(cands)


def _shell_invocation(command: str) -> tuple[list[str] | str, bool] | None:
    """Pick how to run `command` for the current platform → (args, shell) or None.

    POSIX: keep ``shell=True`` (the default ``/bin/sh``).
    Windows: the model emits unix commands (ls/grep/find/rm/python3) that cmd.exe can't run, so we
    require a POSIX ``bash`` (Git-Bash or WSL) and invoke it explicitly as ``bash -c <command>``.
    Returns None when no bash is found → caller reports an actionable error.
    """
    if not _is_windows():
        return command, True
    bash = _find_bash()
    if bash:
        return [bash, "-c", command], False
    return None


def _shell_env() -> dict[str, str] | None:
    """Subprocess environment. On Windows force a UTF-8 locale so Git-Bash (MSYS2) emits filenames as
    UTF-8 — otherwise `ls`/`find`/… output non-ASCII names (CJK/日本語/한글/…) in the system codepage
    (cp950/cp936/cp932…), which we'd then decode as garbage. None on POSIX = inherit env unchanged."""
    if not _is_windows():
        return None
    return {**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}


_HEALTH_MARKER = "__falamus_shell_ok__"


def shell_healthcheck() -> tuple[bool, str]:
    """Verify run_command will actually work. POSIX: always ok. Windows: find a bash AND RUN it, so a bash
    that exists but is broken is caught (e.g. the System32 WSL launcher with no distro installed — it
    'exists' but every command fails/hangs). Returns (ok, detail): detail = the bash path when ok, or a
    user-facing problem message when not."""
    if not _is_windows():
        return True, ""
    bash = _find_bash()
    if not bash:
        return False, ("No bash found. Shell commands (run_command) won't work — install Git for Windows "
                       "(Git-Bash, https://gitforwindows.org) or WSL, then restart.")
    try:
        proc = subprocess.run(
            [bash, "-c", f"echo {_HEALTH_MARKER}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=_shell_env(), timeout=10,
        )
    except Exception as e:
        return False, f"bash was found ({bash}) but could not run ({type(e).__name__}). Install Git for Windows."
    if _HEALTH_MARKER in (proc.stdout or ""):
        return True, bash
    return False, (f"bash was found ({bash}) but isn't working — most likely the WSL launcher with no "
                   "Linux distro installed. Install Git for Windows (Git-Bash) so shell commands work.")


def _clip(s: str) -> str:
    if len(s) <= _MAX_OUTPUT:
        return s
    head = s[: _MAX_OUTPUT // 2]
    tail = s[-_MAX_OUTPUT // 2:]
    return f"{head}\n…[{len(s) - _MAX_OUTPUT} chars omitted]…\n{tail}"


def make_tools(workdir: str | None = None) -> list[Tool]:
    base = Path(workdir).expanduser().resolve() if workdir else Path.cwd()

    def run_command(args: dict) -> ToolResult:
        command = args["command"]
        timeout = int(args.get("timeout", _DEFAULT_TIMEOUT))
        invocation = _shell_invocation(command)
        if invocation is None:
            return ToolResult.error(
                "No POSIX shell found. falamus runs shell commands through bash; on Windows install "
                "Git for Windows (Git-Bash) or WSL so a `bash` is on PATH, then retry."
            )
        cmd_args, use_shell = invocation
        try:
            proc = subprocess.run(
                cmd_args,
                shell=use_shell,
                cwd=str(base),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",   # safety net: any still-undecodable byte (e.g. localized output from a
                                    # native Windows .exe that ignores LC_ALL) becomes U+FFFD instead of
                                    # crashing the subprocess reader thread (which would garble the TUI).
                env=_shell_env(),   # Windows: force UTF-8 locale so Git-Bash emits filenames as UTF-8.
                timeout=timeout,
                stdin=subprocess.DEVNULL,   # interactive input() gets EOF immediately → fails fast instead of hanging
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error(
                f"Command timed out ({timeout}s): {command}. "
                "If this is an interactive program that waits for input, do NOT run it directly to test; "
                "use a syntax check (e.g. python3 -m py_compile) or pipe input (e.g. echo 50 | python3 prog.py)."
            )
        parts = [f"$ {command}", f"(exit={proc.returncode})"]
        if proc.stdout:
            parts.append("--- stdout ---\n" + _clip(proc.stdout.rstrip()))
        if proc.stderr:
            parts.append("--- stderr ---\n" + _clip(proc.stderr.rstrip()))
        return ToolResult(text="\n".join(parts), is_error=proc.returncode != 0)

    return [
        Tool(
            name="run_command",
            description=frag("tools", "run_command"),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the shell command to run"},
                    "timeout": {"type": "integer", "description": "timeout in seconds (default 60)"},
                },
                "required": ["command"],
            },
            handler=run_command,
            risk="high",
        ),
    ]
