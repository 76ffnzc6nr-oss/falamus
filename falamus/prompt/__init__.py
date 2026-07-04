"""Single source of truth for model-facing prompt fragments — now as swappable prompt SETS.

A prompt SET is a folder holding the category `.md` files (rules.md, agents.md, tools.md, notes.md), each
with several `## <name>` fragments (a leading ``<!-- scope: … -->`` comment documents the fragment and is
stripped before the model sees it). Sets live either built-in (`falamus/prompt/<set>/`) or user-authored
(`~/.config/falamus/prompts/<set>/` — copy a built-in one and edit). The ACTIVE set is chosen ONCE at
startup by the backend (local vs cloud → `set_active(name)`); `frag()` reads from it.

`default_local` = the full weak-model scaffolding (personas + working rules + tool contracts). `default_cloud`
= minimal (a capable model doesn't need the persona/rules — those fragments are blank; only the tool
contracts and a few falamus-specific notes remain). Users add sets by copying either and editing.

To change a prompt, edit the `.md` fragment — don't hand-copy prompt text into code (a golden test locks
`default_local` byte-identical).
"""

from __future__ import annotations

import re
from functools import cache
from importlib.resources import files
from pathlib import Path

_SECTION = re.compile(r"^##\s+(?P<name>\S+)\s*$", re.MULTILINE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

DEFAULT_LOCAL = "default_local"
DEFAULT_CLOUD = "default_cloud"
_BUILTIN = (DEFAULT_LOCAL, DEFAULT_CLOUD)   # shipped, read-only: always resolved from the package (never shadowed)
_CATEGORIES = ("rules", "agents", "tools", "notes")

_active = DEFAULT_LOCAL   # module-global active set; set once at startup (Backend.build), fixed for the session


def set_active(name: str | None) -> None:
    """Select the active prompt set for this session (called by the backend). Empty → default_local."""
    global _active
    _active = name or DEFAULT_LOCAL


def active() -> str:
    return _active


def user_prompts_dir() -> Path:
    """Where user-authored sets live (alongside config.ini): ~/.config/falamus/prompts/ (or %APPDATA%)."""
    from falamus.settings import CONFIG_PATH
    return CONFIG_PATH.parent / "prompts"


def _category_text(set_name: str, category: str) -> str:
    """Read a set's category .md. A BUILT-IN set is always read from the package (shipped, read-only — so a
    package update reaches everyone and the golden locks the real source); any OTHER name is a user-added set
    read from the config dir. To customise a built-in, copy it to a new name (`--copy-prompt`)."""
    fname = f"{category}.md"
    if set_name in _BUILTIN:
        return (files(__package__) / set_name / fname).read_text(encoding="utf-8")
    return (user_prompts_dir() / set_name / fname).read_text(encoding="utf-8")


@cache
def _category(set_name: str, category: str) -> dict[str, str]:
    """Parse `<set>/<category>.md` into {section_name: body_text} (scope comments stripped). Cached per set."""
    text = _category_text(set_name, category)
    out: dict[str, str] = {}
    matches = list(_SECTION.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end]
        body = _COMMENT.sub("", body).strip()
        out[m.group("name")] = body
    return out


def frag(category: str, name: str) -> str:
    """Return one prompt fragment (verbatim body text) by category + name, from the ACTIVE set."""
    try:
        return _category(_active, category)[name]
    except KeyError as e:
        raise KeyError(f"prompt fragment not found: {_active}/{category}/{name}") from e


def fragments(category: str) -> dict[str, str]:
    """All fragments in a category of the active set (read-only copy)."""
    return dict(_category(_active, category))


def _is_set(path) -> bool:
    """A directory is a prompt set if it holds every category file."""
    try:
        return all((path / f"{c}.md").is_file() for c in _CATEGORIES)
    except (OSError, AttributeError):
        return False


def copy_builtin(builtin: str, new_name: str) -> Path:
    """Copy a built-in set into the user dir under a NEW name, so it can be edited (built-ins are read-only,
    read straight from the package). Returns the destination dir. Raises if `builtin` isn't a built-in or the
    destination already exists."""
    if builtin not in _BUILTIN:
        raise ValueError(f"not a built-in set: {builtin!r} (choose from {_BUILTIN})")
    if new_name in _BUILTIN:
        raise ValueError(f"{new_name!r} is a built-in name; pick a different name")
    dst = user_prompts_dir() / new_name
    if dst.exists():
        raise ValueError(f"prompt set already exists: {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    for cat in _CATEGORIES:
        (dst / f"{cat}.md").write_text(
            (files(__package__) / builtin / f"{cat}.md").read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def list_sets() -> list[str]:
    """Selectable set names: the BUILT-INS (from the package) always first, then any USER-ADDED sets found in
    ~/.config/falamus/prompts/ (a folder whose name isn't a built-in). Sorted, built-ins first."""
    names = list(_BUILTIN)
    udir = user_prompts_dir()
    if udir.is_dir():
        for p in sorted(udir.iterdir()):
            if p.is_dir() and p.name not in names and _is_set(p):
                names.append(p.name)
    return names
