"""Persistence of every agent's context.

Persists each agent's full message history and the session manifest to disk, supporting:
  - step-by-step checkpoints (overwritten each step; at most the last step is lost on crash).
  - resume after crash / machine switch: restore the main agent's conversation and continue.

File layout (inside the project, moves with it):
  <workdir>/.falamus/sessions/<sid>/
    ├── meta.json            # manifest: sid / workdir / backend config / agent states
    ├── agents/<name>.json   # each agent's message-history checkpoint
    └── artifacts/<id>/      # sub-agent delivery folders

Uses atomic writes (write .tmp then rename) to avoid corruption from a mid-write crash.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .workspace import session_base


class SessionStore:
    def __init__(self, session_root: str | Path) -> None:
        self.root = Path(session_root)
        self.agents_dir = self.root / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)

    # ---- atomic write ---------------------------------------------------
    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    # ---- agent checkpoints ----------------------------------------------
    def save_agent(self, name: str, messages: list[dict[str, Any]]) -> None:
        payload = {"name": name, "updated": time.time(), "messages": messages}
        self._atomic_write(
            self.agents_dir / f"{name}.json",
            json.dumps(payload, ensure_ascii=False),
        )

    def load_agent(self, name: str) -> list[dict[str, Any]] | None:
        p = self.agents_dir / f"{name}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("messages")
        except (json.JSONDecodeError, OSError):
            return None

    def agent_names(self) -> list[str]:
        return sorted(p.stem for p in self.agents_dir.glob("*.json"))

    # ---- per-agent memo (external scratchpad; NOT injected into the prompt) ----
    # Each agent keeps its own private to-do list here, outside the conversation, so the prompt prefix
    # never mutates (the KV cache stays valid). The `memo` tool reads/writes these files on demand.
    def load_memo(self, name: str) -> str:
        p = self.root / "memos" / f"{name}.md"
        try:
            return p.read_text(encoding="utf-8") if p.exists() else ""
        except OSError:
            return ""

    def save_memo(self, name: str, text: str) -> None:
        (self.root / "memos").mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.root / "memos" / f"{name}.md", text)

    # ---- manifest -------------------------------------------------------
    def save_manifest(self, meta: dict[str, Any]) -> None:
        meta = {**meta, "updated": time.time()}
        self._atomic_write(self.root / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

    def load_manifest(self) -> dict[str, Any]:
        p = self.root / "meta.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}


def list_sessions(workdir: str | Path | None = None) -> list[dict[str, Any]]:
    """List all sessions under a working directory (newest first), with manifest summaries."""
    base = session_base(workdir)
    out: list[dict[str, Any]] = []
    for d in base.iterdir() if base.exists() else []:
        if not d.is_dir():
            continue
        store = SessionStore(d)
        meta = store.load_manifest()
        out.append({
            "sid": d.name,
            "updated": meta.get("updated", d.stat().st_mtime),
            "agents": meta.get("agents", []),
            "title": meta.get("title", ""),
            "path": str(d),
        })
    out.sort(key=lambda x: x["updated"], reverse=True)
    return out


if __name__ == "__main__":
    import tempfile

    wd = tempfile.mkdtemp()
    root = session_base(wd) / "20260619-test"
    store = SessionStore(root)
    store.save_agent("main", [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}])
    store.save_manifest({"sid": "20260619-test", "workdir": wd,
                         "backend": "llama_cpp", "title": "test task",
                         "agents": [{"id": "a1", "status": "done"}]})
    print("load_agent main:", store.load_agent("main"))
    print("manifest:", store.load_manifest().get("title"))
    print("list_sessions:", [s["sid"] for s in list_sessions(wd)])
