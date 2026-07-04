"""Centralized error log.

Any error during a helper run (agent/sub-agent tool errors, circuit breaker, iteration
limit, exceptions, ...) is written, tagged with its source (which agent), into
<workdir>/.falamus/error_log.md, for later analysis to improve prompts / architecture.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .workspace import helper_dir

_HEADER = (
    "# helper error log\n\n"
    "> Automatically records runtime errors (agent/sub-agent/tool/circuit-breaker/exception), "
    "for later prompt and architecture improvements.\n"
)
_MAX = 2000   # truncation cap per message/detail


class ErrorLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    @classmethod
    def for_workdir(cls, workdir: str | Path | None) -> ErrorLog:
        return cls(helper_dir(workdir) / "error_log.md")

    def log(self, source: str, kind: str, message: str, detail: str = "") -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = [f"\n## {ts} — [{source}] {kind}", str(message)[:_MAX].rstrip()]
        if detail:
            entry.append(f"\n```\n{str(detail)[:_MAX].rstrip()}\n```")
        block = "\n".join(entry) + "\n"
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if not self.path.exists():
                    self.path.write_text(_HEADER, encoding="utf-8")
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(block)
        except OSError:
            pass   # a logging failure must not affect the main flow


if __name__ == "__main__":
    import tempfile
    el = ErrorLog(Path(tempfile.mkdtemp()) / "error_log.md")
    el.log("sub_1", "tool_error", "run_command failed", "exit=1 stderr: command not found")
    el.log("main", "spawn_failed", "sub-agent consecutive failures reached the limit")
    print(el.path.read_text())
