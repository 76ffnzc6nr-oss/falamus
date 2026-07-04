"""Project rules file falamus.md.

On startup:
  - Check whether the working directory has falamus.md; if not, create a template.
  - If it exists, read it in and inject it into the main agent's system prompt
    (i.e. "project rules").
"""

from __future__ import annotations

from pathlib import Path

from falamus.prompt import frag

RULES_FILENAME = "falamus.md"

# ── Working-rules building blocks. The text lives ONCE in the active prompt set's rules.md; this module
#    composes the per-role buckets from those fragments AT RUNTIME (functions, not import-time constants) so
#    the active set — local vs cloud — is honoured. For `default_local` the composition is byte-identical to
#    the pre-refactor text (locked by the golden test). A / B / C / D roles below. ──
def common_rules() -> str:
    """A — COMMON: every agent with file tools sees these (main + sub)."""
    return frag("rules", "common")


def main_rules() -> str:
    """B — main agent (delegate + memo)."""
    return frag("rules", "delegate") + "\n" + frag("rules", "memo")


def sub_rules() -> str:
    """C — a sub-agent that can still spawn (delegate + path + memo)."""
    return frag("rules", "delegate") + "\n" + frag("rules", "path") + "\n" + frag("rules", "memo")


def last_sub_rules() -> str:
    """D — a leaf sub-agent (path only; single-shot, so no memo)."""
    return frag("rules", "path")


def working_rules(role_rules: str) -> str:
    """Compose a '## Working rules' block from a role bucket + the COMMON rules (single source of COMMON).
    Blank on both (a minimal/cloud set) → no empty header at all."""
    common = common_rules()
    if not role_rules.strip() and not common.strip():
        return ""
    return "## Working rules\n" + role_rules + "\n" + common


def template() -> str:
    """The falamus.md template for the active set (blank rule fragments → a minimal template, e.g. cloud)."""
    return f"""# falamus project rules

> Long-term notes for falamus in this project. Read automatically on startup.
> Edit freely (bullet points); no restart needed.
> (Reply language is controlled by the app's language setting, not this file.)

## Project background
- (describe what this project is and its goal)

{working_rules(main_rules())}

## Forbidden
- (list things not to do, e.g. do not delete the xxx directory)
"""


def load_or_create_rules(workdir: str | Path | None = None) -> tuple[str, bool]:
    """Return (rules_text, created_new).

    workdir defaults to the current working directory.
    """
    base = Path(workdir).expanduser().resolve() if workdir else Path.cwd()
    p = base / RULES_FILENAME
    if p.exists():
        return p.read_text(encoding="utf-8"), False
    text = template()
    p.write_text(text, encoding="utf-8")
    return text, True


def read_rules(workdir: str | Path | None = None) -> str:
    """Return the on-disk falamus.md content, or '' if it doesn't exist. NEVER creates it (unlike
    load_or_create_rules) — used by plain-chat mode, which injects an existing rules file but must not
    seed the tool-version template."""
    base = Path(workdir).expanduser().resolve() if workdir else Path.cwd()
    p = base / RULES_FILENAME
    return p.read_text(encoding="utf-8") if p.exists() else ""


# falamus-managed "last progress" block (only this block is overwritten; user content untouched)
PROGRESS_BEGIN = "<!-- falamus:last-progress -->"
PROGRESS_END = "<!-- /falamus:last-progress -->"
_PROGRESS_TITLE = "## Last progress (auto-maintained by falamus — do not edit)"


def update_last_progress(workdir: str | Path | None, summary: str) -> Path:
    """Write a conversation summary into falamus.md's managed block (only that block).

    Written directly by the program (not via a model tool), so it never triggers a safety prompt.
    """
    base = Path(workdir).expanduser().resolve() if workdir else Path.cwd()
    p = base / RULES_FILENAME
    # If falamus.md doesn't exist (e.g. plain-chat mode never created one), DON'T seed the tool-version
    # _TEMPLATE just to store a summary — that would write tool Working rules into a tool-less session.
    # Start from empty so the file holds only the progress block.
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    block = f"{PROGRESS_BEGIN}\n{_PROGRESS_TITLE}\n{summary.strip()}\n{PROGRESS_END}"
    if PROGRESS_BEGIN in text and PROGRESS_END in text:
        pre = text.split(PROGRESS_BEGIN, 1)[0].rstrip()
        post = text.split(PROGRESS_END, 1)[1].lstrip()
        text = (pre + "\n\n" + block + ("\n\n" + post if post else "\n")).rstrip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    p.write_text(text, encoding="utf-8")
    return p


def inject_into_prompt(base_prompt: str, rules_text: str) -> str:
    """Merge the rules text into the system prompt."""
    if not rules_text.strip():
        return base_prompt
    return (
        base_prompt
        + "\n\n" + frag("notes", "rules_header") + "\n"
        + rules_text.strip()
    )


if __name__ == "__main__":
    import tempfile

    d = tempfile.mkdtemp()
    text, created = load_or_create_rules(d)
    print("first:", "created" if created else "read", "| length", len(text))
    text2, created2 = load_or_create_rules(d)
    print("again:", "created" if created2 else "read")
    print("\nExample injected system prompt (excerpt):")
    print(inject_into_prompt("You are the main agent.", text2)[:200])
