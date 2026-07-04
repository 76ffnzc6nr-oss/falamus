"""Working directory (project root) location.

The "working directory" = the project root workdir: all file/CLI/image tools are based
on it. By the user's choice, sub-agent delivery folders / sessions / checkpoints live
**inside the project**:
    <workdir>/.falamus/sessions/<sid>/artifacts/<agent_id>/

A .gitignore (content `*`) is written automatically under <workdir>/.falamus/ so these
intermediate artifacts stay out of version control and don't pollute the project.
"""

from __future__ import annotations

from pathlib import Path


def resolve_workdir(workdir: str | Path | None = None) -> Path:
    """Normalize the project root; defaults to the current working directory."""
    return Path(workdir).expanduser().resolve() if workdir else Path.cwd()


def helper_dir(workdir: str | Path | None = None) -> Path:
    """Get <workdir>/.falamus, ensuring it exists and is gitignored."""
    base = resolve_workdir(workdir) / ".falamus"
    base.mkdir(parents=True, exist_ok=True)
    gi = base / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n", encoding="utf-8")  # everything under .falamus stays out of VCS
    return base


def session_base(workdir: str | Path | None = None) -> Path:
    """Session root: <workdir>/.falamus/sessions."""
    d = helper_dir(workdir) / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    import tempfile

    d = tempfile.mkdtemp()
    sb = session_base(d)
    print("workdir     :", resolve_workdir(d))
    print("session_base:", sb)
    print(".gitignore content:", (helper_dir(d) / ".gitignore").read_text().strip())
