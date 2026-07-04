"""Bottom status bar.

Uses an ANSI scroll region (DECSTBM) to pin the terminal's last line to the bottom as a status bar:
  - enable: set the scroll region to "line 1 ~ second-to-last line"; normal output scrolls there;
            the last line is outside the region, fixed as the status bar.
  - set(): repaint the status text in place on the last line (save/restore cursor, doesn't disturb typing).
  - terminal resize (SIGWINCH): recompute size, reset the scroll region, repaint (else it vanishes after resize).
  - disable: restore the scroll region.

Degrades to a no-op on non-TTY (no control codes when piped/redirected). No third-party dependencies.
"""

from __future__ import annotations

import shutil
import signal
import sys
from typing import Any, TextIO


class StatusBar:
    def __init__(self, stream: TextIO | None = None, enabled: bool | None = None) -> None:
        self.stream = stream or sys.stdout
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self._text = ""
        self._active = False           # don't draw before enable() to avoid startup residue
        self._prev_winch: Any = None        # previous SIGWINCH handler

    def _size(self) -> tuple[int, int]:
        sz = shutil.get_terminal_size((80, 24))
        return sz.columns, sz.lines

    def enable(self) -> None:
        if not self.enabled:
            return
        _cols, rows = self._size()
        # reserve one bottom line: print a newline to push the cursor down, set the scroll region, move to its bottom
        self.stream.write("\n")
        self.stream.write(f"\033[1;{rows - 1}r")
        self.stream.write(f"\033[{rows - 1};1H")
        self.stream.flush()
        self._active = True
        self._draw(set_region=False)
        # listen for terminal resize
        try:
            self._prev_winch = signal.signal(signal.SIGWINCH, self._on_resize)
        except (ValueError, OSError, AttributeError):
            self._prev_winch = None     # not the main thread / platform unsupported → skip

    def set(self, text: str) -> None:
        self._text = text
        self._draw(set_region=False)

    def _draw(self, set_region: bool) -> None:
        """Draw the bottom status bar. set_region=True also resets the scroll region (used after resize)."""
        if not (self.enabled and self._active):
            return
        cols, rows = self._size()
        line = self._text[: cols - 1].ljust(cols - 1)
        self.stream.write("\0337")                       # save cursor
        if set_region:
            self.stream.write(f"\033[1;{rows - 1}r")     # reset scroll region (DECSTBM moves the cursor, so wrap in save/restore)
        self.stream.write(f"\033[{rows};1H\033[2K")      # move to bottom line, clear it
        self.stream.write("\033[7m" + line + "\033[0m")  # print inverted
        self.stream.write("\0338")                       # restore cursor
        self.stream.flush()

    def _on_resize(self, signum, frame) -> None:
        """Terminal resize: re-layout so the old status bar isn't reflowed into the scroll region as residue.

        Steps: restore scroll region → clear screen → set new scroll region → repaint the bottom bar →
        chain to readline (so it reprints the current input line on a clean screen).
        Cleared conversation stays in the terminal scrollback and can be scrolled up to view.
        """
        if self.enabled and self._active:
            _cols, rows = self._size()
            self.stream.write("\033[r")                  # restore scroll region (full page)
            self.stream.write("\033[2J\033[H")           # clear screen + home cursor
            self.stream.write(f"\033[1;{rows - 1}r")     # set new scroll region
            self.stream.write(f"\033[{rows - 1};1H")     # move cursor to the bottom of the region
            self.stream.flush()
            self._draw(set_region=False)                 # repaint the bar at the new bottom
        if callable(self._prev_winch):                   # chain to readline: reprint the input line
            try:
                self._prev_winch(signum, frame)
            except Exception:
                pass

    def disable(self) -> None:
        if not self.enabled or not self._active:
            return
        self._active = False
        if self._prev_winch is not None:
            try:
                signal.signal(signal.SIGWINCH, self._prev_winch)
            except (ValueError, OSError):
                pass
            self._prev_winch = None
        _cols, rows = self._size()
        self.stream.write("\033[r")                      # restore scroll region (whole screen)
        self.stream.write(f"\033[{rows};1H\033[2K")      # clear the bottom status bar
        self.stream.flush()

    def __enter__(self) -> StatusBar:
        self.enable()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disable()
