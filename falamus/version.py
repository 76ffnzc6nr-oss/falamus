"""Single source of truth for the version number.

Everything (TUI status bar, /version, run.py --version, pyproject) reads from here.
Bump per the rules in docs/優化企劃書.md §5 (SemVer: MAJOR.MINOR.PATCH).
"""

__version__ = "1.0.0"
