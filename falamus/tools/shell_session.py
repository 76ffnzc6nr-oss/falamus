"""Persistent interactive shell sessions (PTY) — opt-in, POSIX-only.

Unlike `run_command` (one command → wait → done), a shell session stays alive so the agent can interact
back and forth with a long-running program: a REPL, an installer that prompts, or a little game that
reads stdin and prints to stdout. Three tools drive it:

  - `shell_open(command, note)`   : start a program under a PTY, return its pid + first output.
  - `shell_input(handle, input?)` : send one line (omit `input` to just poll for new output), read back.
  - `shell_close(handle)`         : terminate it.

Lifetime is bounded to a SINGLE conversation turn: the backend closes every session in run_message's
`finally`, so nothing survives between user messages (no cross-turn leak). A process-group kill
(`pty.fork` makes each child a session leader) takes the program AND its children down cleanly.

POSIX only: a PTY needs the stdlib `pty` module. On Windows these tools are simply not offered (keeps
falamus pure-python / dependency-free — no pywinpty). Reading uses a quiet-period heuristic: a program
that stops emitting output for `_QUIET` seconds is taken to be waiting for input, so the read returns
then (interactive prompts deliberately end WITHOUT a newline, so "ends with newline" is the wrong
terminator — silence is the right one), bounded by a hard `_MAX_TOTAL` cap.
"""

from __future__ import annotations

import os
import re
import signal
import threading
import time
from dataclasses import dataclass, field

from .registry import Tool, ToolResult

SHELL_TOOLS = ["shell_open", "shell_input", "shell_close"]

# read-timing knobs (seconds)
_FIRST_TIMEOUT = 3.0    # how long to wait for the FIRST byte before giving up on this read (a silent program
                        # waiting for input returns nothing fast; if output is just slow, poll again with
                        # shell_input and no 'input' — so this need not be long)
_QUIET = 0.4            # once output goes quiet this long, the program is taken to be waiting → return
_MAX_TOTAL = 30.0       # hard cap on a single read (a chatty/looping program can't hang the turn)
_POLL = 0.05           # buffer poll interval
_GRACEFUL = 0.6        # after sending EOF, how long to let a program exit CLEANLY before signalling it

# per-owner and global open-session limits (an agent that hits its limit must close one first)
_MAX_PER_AGENT = 1
_MAX_TOTAL_OPEN = 4

# hard cap on a session's UNREAD buffer (bytes). Far above any normal interactive output, so normal use
# never trims; a flooding program (e.g. `yes`, `cat /dev/urandom`) is capped here instead of growing the
# buffer without bound until OOM. Once full the reader drops the NEWEST bytes (keeps the front stable so a
# concurrent read's slice stays valid) and flags truncation; the agent is told its output was capped.
_MAX_BUFFER = 4_000_000

_MAX_OUTPUT = 30_000   # clip a single read's text (head + tail), same budget as run_command

# strip terminal escapes so a weak model sees plain text: CSI (colour/cursor) + OSC (title) sequences
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def available() -> bool:
    """Whether persistent shells can run here (POSIX with a usable pty). False on Windows."""
    return os.name != "nt"


def _clean(raw: bytes) -> str:
    """Decode + de-escape PTY output into plain text a model can read."""
    text = raw.decode("utf-8", errors="replace")
    text = _ANSI.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > _MAX_OUTPUT:
        head, tail = text[: _MAX_OUTPUT // 2], text[-_MAX_OUTPUT // 2:]
        text = f"{head}\n…[{len(text) - _MAX_OUTPUT} chars omitted]…\n{tail}"
    return text


@dataclass
class _Session:
    pid: int
    fd: int
    owner: str
    note: str
    command: str
    buf: bytearray = field(default_factory=bytearray)   # UNREAD output only — each read drains + trims it
    truncated: bool = False   # set when a flood hit _MAX_BUFFER and bytes were dropped (reported to the agent)
    lock: threading.Lock = field(default_factory=threading.Lock)
    alive: bool = True
    exit_code: int | None = None
    reader: threading.Thread | None = None

    def _read_loop(self) -> None:
        while True:
            try:
                data = os.read(self.fd, 4096)
            except OSError:
                data = b""          # EIO on Linux when the child exits == EOF
            if not data:
                break
            with self.lock:
                room = _MAX_BUFFER - len(self.buf)
                if room <= 0:                       # buffer full → drop NEWEST (front stays stable for readers)
                    self.truncated = True
                else:
                    self.buf.extend(data[:room])
                    if len(data) > room:
                        self.truncated = True       # filled to the cap; the rest is dropped
        self._reap()

    def _reap(self) -> None:
        with self.lock:
            self.alive = False
        try:
            _pid, status = os.waitpid(self.pid, 0)
            self.exit_code = os.waitstatus_to_exitcode(status)
        except OSError:
            pass


class ShellManager:
    """Holds the live shell sessions for one runtime (one per agent-owner, capped). Thread-safe."""

    def __init__(self) -> None:
        self._sessions: dict[int, _Session] = {}
        self._lock = threading.Lock()

    # ---- introspection (status bar / user) ------------------------------
    def open_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def listing(self) -> list[tuple[int, str, str]]:
        """(pid, owner, note) for every open session — for the user / status."""
        with self._lock:
            return [(s.pid, s.owner, s.note) for s in self._sessions.values()]

    # ---- lifecycle ------------------------------------------------------
    def open(self, owner: str, command: str, note: str, cwd: str) -> ToolResult:
        if not available():
            return ToolResult.error("persistent shell sessions are not available on this platform (POSIX only).")
        with self._lock:
            mine = sum(1 for s in self._sessions.values() if s.owner == owner)
            if mine >= _MAX_PER_AGENT:
                return ToolResult.error(
                    f"you already have {mine} open shell session(s) (limit {_MAX_PER_AGENT}). "
                    "shell_close the existing one before opening another.")
            if len(self._sessions) >= _MAX_TOTAL_OPEN:
                return ToolResult.error(
                    f"the global open-shell limit ({_MAX_TOTAL_OPEN}) is reached. Close one first.")
        import pty
        pid, fd = pty.fork()
        if pid == 0:   # child: become the program (pty.fork already did setsid + controlling tty)
            try:
                os.chdir(cwd)
            except OSError:
                pass
            env = {**os.environ, "TERM": "dumb", "PYTHONUNBUFFERED": "1"}
            os.execvpe("/bin/sh", ["/bin/sh", "-c", command], env)
            os._exit(127)   # exec failed
        sess = _Session(pid=pid, fd=fd, owner=owner, note=note or "", command=command)
        sess.reader = threading.Thread(target=sess._read_loop, daemon=True)
        sess.reader.start()
        with self._lock:
            self._sessions[pid] = sess
        out = self._read_new(sess, sent=False)
        head = (f"shell session opened (pid {pid}): {command}\n"
                f"(note: {note})\n" if note else f"shell session opened (pid {pid}): {command}\n")
        body = out or "(no output yet)"
        tail = self._status_tail(sess)
        return ToolResult(text=head + "--- output ---\n" + body + tail)

    def input(self, owner: str, handle: int | str, text: str | None) -> ToolResult:
        sess = self._owned(owner, handle)
        if isinstance(sess, ToolResult):
            return sess
        if not sess.alive:
            return ToolResult.error(
                f"shell session {sess.pid} has already exited (exit code {sess.exit_code}). "
                "shell_open a new one if you need to continue.")
        if text is not None:
            try:
                os.write(sess.fd, (text + "\n").encode("utf-8"))
            except OSError as e:
                return ToolResult.error(f"could not write to shell {sess.pid}: {e}")
        out = self._read_new(sess, sent=text is not None, echo=text)
        body = out or "(no new output)"
        return ToolResult(text="--- output ---\n" + body + self._status_tail(sess))

    def close(self, owner: str, handle: int | str) -> ToolResult:
        sess = self._owned(owner, handle)
        if isinstance(sess, ToolResult):
            return sess
        self._terminate(sess)
        with self._lock:
            self._sessions.pop(sess.pid, None)
        return ToolResult(text=f"shell session {sess.pid} closed (exit code {sess.exit_code}).")

    # ---- user-initiated /kill <pid> -------------------------------------
    def kill(self, pid: int) -> str:
        """Force-kill a falamus-owned session by pid (user command). Refuses unknown pids."""
        with self._lock:
            sess = self._sessions.get(pid)
        if sess is None:
            return f"no falamus shell session with pid {pid} (only sessions falamus started can be killed)."
        self._terminate(sess)
        with self._lock:
            self._sessions.pop(pid, None)
        return f"killed shell session {pid} ({sess.note or sess.command})."

    def close_all(self) -> None:
        """Close every session — called at the end of each turn and at exit (idempotent)."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for sess in sessions:
            self._terminate(sess)

    # ---- internals ------------------------------------------------------
    def _owned(self, owner: str, handle: int | str) -> _Session | ToolResult:
        try:
            pid = int(handle)
        except (TypeError, ValueError):
            return ToolResult.error(f"invalid shell handle: {handle!r} (use the pid from shell_open).")
        with self._lock:
            sess = self._sessions.get(pid)
        if sess is None:
            return ToolResult.error(f"no open shell session with handle {pid} (it may have been closed).")
        if sess.owner != owner:
            return ToolResult.error(f"shell session {pid} belongs to another agent; you can only use your own.")
        return sess

    def _terminate(self, sess: _Session) -> None:
        # GRACEFUL FIRST: send EOF (Ctrl-D) so a stdin-reading program ends its own loop and exits cleanly
        # (its atexit / cleanup handlers run, exit code 0) instead of being signalled. Escalate to a whole-
        # process-group SIGTERM→SIGKILL only if it doesn't exit in time. (Reported by a live test as
        # shell_close leaving exit code -15 with no chance to clean up.)
        if sess.alive and sess.fd >= 0:
            try:
                os.write(sess.fd, b"\x04")
            except OSError:
                pass
            if not self._wait_exit(sess, _GRACEFUL):
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(sess.pid, sig)   # whole process group → also reaps spawned children
                    except OSError:
                        break
                    if self._wait_exit(sess, 1.0):
                        break
        if sess.fd >= 0:
            try:
                os.close(sess.fd)
            except OSError:
                pass
            sess.fd = -1
        if sess.reader is not None:
            sess.reader.join(timeout=1.0)

    @staticmethod
    def _wait_exit(sess: _Session, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not sess.alive:
                return True
            time.sleep(_POLL)
        return not sess.alive

    @staticmethod
    def _status_tail(sess: _Session) -> str:
        if not sess.alive:
            return f"\n(session ended; exit code {sess.exit_code})"
        return "\n(session still running; the program appears to be waiting for input)"

    @staticmethod
    def _read_new(sess: _Session, sent: bool, echo: str | None = None) -> str:
        """Drain output not yet returned to the agent, using the quiet-period heuristic + a hard cap.

        sess.buf holds ONLY unread output: each call returns everything in it and removes it (del buf[:end]).
        Output that landed in the gap between two reads is therefore never skipped — it is still sitting in
        buf and returned by the next read (lossless). Trimming what was returned keeps memory bounded; a
        flooding program is additionally capped by _MAX_BUFFER in the reader (sess.truncated → noted here).
        """
        with sess.lock:
            cur0 = len(sess.buf)
        t0 = time.monotonic()
        last_len = cur0
        last_change = t0
        # pre-existing unread bytes (arrived in the gap before this call) count as output already
        got_any = cur0 > 0
        while True:
            time.sleep(_POLL)
            now = time.monotonic()
            with sess.lock:
                cur = len(sess.buf)
                alive = sess.alive
            if cur > last_len:
                last_len = cur
                last_change = now
                got_any = True
            if not alive:
                break
            if got_any and (now - last_change) >= _QUIET:
                break
            if not got_any and (now - t0) >= _FIRST_TIMEOUT:
                break
            if (now - t0) >= _MAX_TOTAL:
                break
        with sess.lock:
            raw = bytes(sess.buf)        # all unread bytes
            del sess.buf[:]              # consumed → drop them (bounded memory; lossless: only unread remained)
            capped = sess.truncated
            sess.truncated = False
        text = _clean(raw)
        if capped:
            text += "\n…[output truncated: the program is flooding output; it was capped to protect memory]…"
        # drop the tty's echo of the line we just typed (it comes back as the first line)
        if echo:
            lines = text.split("\n")
            if lines and lines[0].strip() == echo.strip():
                text = "\n".join(lines[1:])
        return text.strip()


def make_shell_tools(manager: ShellManager, owner: str, cwd: str) -> list[Tool]:
    """Build the three shell-session tools bound to a manager + owning agent + working directory."""
    from falamus.prompt import frag

    def _open(args: dict) -> ToolResult:
        return manager.open(owner, args["command"], args.get("note", ""), cwd)

    def _input(args: dict) -> ToolResult:
        return manager.input(owner, args["handle"], args.get("input"))

    def _close(args: dict) -> ToolResult:
        return manager.close(owner, args["handle"])

    return [
        Tool(
            name="shell_open",
            description=frag("tools", "shell_open"),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the program/command to launch (e.g. python3 game.py)"},
                    "note": {"type": "string", "description": "a short note describing what this session is for (shown to the user)"},
                },
                "required": ["command"],
            },
            handler=_open,
            risk="high",
        ),
        Tool(
            name="shell_input",
            description=frag("tools", "shell_input"),
            parameters={
                "type": "object",
                "properties": {
                    "handle": {"type": "integer", "description": "the session pid returned by shell_open"},
                    "input": {"type": "string", "description": "one line to send to the program; OMIT to just read new output"},
                },
                "required": ["handle"],
            },
            handler=_input,
            risk="high",
        ),
        Tool(
            name="shell_close",
            description=frag("tools", "shell_close"),
            parameters={
                "type": "object",
                "properties": {
                    "handle": {"type": "integer", "description": "the session pid returned by shell_open"},
                },
                "required": ["handle"],
            },
            handler=_close,
            risk="low",
        ),
    ]
