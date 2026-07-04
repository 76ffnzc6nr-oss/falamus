"""Full-screen TUI (prompt_toolkit).

Layout (top → bottom):
  - Combined output area: all agents' conversation / tool actions; sub-agents distinguished by
    indentation + [id].
  - Status bar: session│workdir│model; the model field takes the leftover width (marquee-scrolls if it
    can't fit) so the other fields are never truncated. Elapsed seconds are always shown — counting while
    busy (with a spinner + "esc to interrupt" hint), frozen at the last turn's duration while idle.
  - Input area: always typeable; you can type even while waiting for the model (send commands / interrupt).

Features:
  - UI decoupled from agents: the task runs on a background thread, the UI never freezes.
  - ESC interrupts the current task (cooperative; takes effect at the next step).
  - Dangerous-action confirmation: answer y/n in the input area.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea

import falamus.i18n as i18n
from falamus.tools.shell_session import SHELL_TOOLS

_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# the "primary argument" to show as the summary for each tool (so spawn_subagent doesn't show as max_tokens=-1)
_PRIMARY_ARG = {
    "spawn_subagent": "task", "run_command": "command", "deliver": "summary",
    "read_file": "path", "write_file": "path", "edit_file": "path",
    "list_dir": "path", "view_image": "path",
}
# tools whose result is shown IN FULL (multi-line) instead of a one-line summary — the persistent shell
# session output, so the user can watch the interaction live in the TUI.
_FULL_RESULT_TOOLS = set(SHELL_TOOLS)

# colors: non-text content (tools/paths/commands/confirmations) distinguished by color and gray
_STYLE = Style.from_dict({
    "status": "reverse",
    "user": "#00afff bold",            # user input
    "assistant": "",                   # main agent reply (default color)
    "assistant.label": "#5fd75f bold",
    "tool": "#808080",                 # tool actions (gray)
    "spawn": "#af87ff bold",           # sub-agent dispatch (purple)
    "done": "#5fd75f",                 # done (green)
    "confirm": "bg:#5f5f5f #ffffff bold",  # safety confirmation (gray bg, prominent)
    "hint": "#ffaf00",                 # commands / hints (orange)
    "path": "#5fafff",                 # paths / files (blue)
    "code": "#d7af5f",                 # `backtick code`
    "error": "#ff5f5f bold",
    "userprompt": "#00afff bold",
    "confirmprompt": "bg:#5f5f5f #ffff00 bold",
    "ctxwarn": "#ffaf00 bold",
    "devmark": "reverse #ffaf00 bold",     # [DEV] marker — its own status field
    "stream": "reverse #5fd75f bold",      # "outputting" live indicator — its own status field
    "cli": "reverse #ffff00 bold",         # persistent shell-session indicator — yellow, its own field
    "working": "reverse #0087ff bold",     # "working" indicator — blue, steady
    "diffadd": "#5fd75f",              # +added (green)
    "diffdel": "#ff5f5f",              # -removed (red)
    "diffhdr": "#00afaf",              # @@ hunk (cyan)
    # scrollbar: gray track, white thumb
    "scrollbar.background": "bg:#5f5f5f",
    "scrollbar.button": "bg:#ffffff",
})

# inline: color `code` and paths (containing /) differently
_INLINE = re.compile(r"(`[^`]+`|(?:[A-Za-z0-9_.\-~]+)?/[^\s,;]+)")


def _inline(text: str, base: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    last = 0
    for m in _INLINE.finditer(text):
        if m.start() > last:
            out.append((base, text[last:m.start()]))
        tok = m.group(0)
        if tok.startswith("`") and tok.endswith("`"):
            out.append(("class:code", tok[1:-1]))
        else:
            out.append(("class:path", tok))
        last = m.end()
    if last < len(text):
        out.append((base, text[last:]))
    return out or [(base, text)]




class _OutputLexer(Lexer):
    """Color each line by its leading marker; also highlight paths and code inline."""

    def lex_document(self, document):
        lines = document.lines

        def get_line(lineno: int):
            text = lines[lineno] if lineno < len(lines) else ""
            s = text.lstrip()
            # diff coloring: +green / -red / @@ cyan
            if s.startswith("@@"):
                return [("class:diffhdr", text)]
            if s.startswith("+") and not s.startswith("+++"):
                return [("class:diffadd", text)]
            if s.startswith("-") and not s.startswith("---"):
                return [("class:diffdel", text)]
            if s.startswith("!"):
                return [("class:confirm", text)]
            if s.startswith("✔") or s.startswith("✘") or s.startswith("⛔"):
                return [("class:hint", text)]
            if text.startswith(i18n.t("prompt")) or s.startswith("you ▷") or s.startswith("你 ▷"):
                return [("class:user", text)]
            if "◀" in text:                       # main agent reply
                i = text.index("◀") + 1
                return [("class:assistant.label", text[:i])] + _inline(text[i:], "class:assistant")
            if s.startswith("⤷"):
                return _inline(text, "class:spawn")
            if s.startswith("·") or s.startswith("↳") or s.startswith("[error]"):
                base = "class:error" if "[error]" in text else "class:tool"
                return _inline(text, base)
            if s.startswith("ok"):
                return _inline(text, "class:done")
            if s.startswith("/") or s.startswith("  /"):
                return [("class:hint", text)]
            return _inline(text, "class:assistant")

        return get_line


class HelperTUI:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        # queue items are tuples: ("line", text) a whole line / ("delta", name, text) a stream fragment
        self._q: asyncio.Queue[tuple] = asyncio.Queue()
        self.busy = False
        self.busy_start = 0.0
        self._elapsed_frozen = 0       # last turn's duration (s); shown while idle so you see how long it took
        self.frame = 0
        self._model_marquee = False    # True when the model name is compressed/scrolling (drives _spin ticks)
        # auto-scroll: follow the newest while _follow is True. It is turned OFF only by a USER scroll-up
        # (Up/PageUp), and back ON by scrolling to the bottom / Ctrl-G / /b / sending a message. While
        # following we FORCE the cursor to the end on every append, so fast streaming can't knock it off.
        self._follow = True
        self._streaming: str | None = None   # name of the agent currently streaming (None = none)
        self._streamed_main = False    # whether the main agent already streamed content this turn (avoid re-printing)
        self._current_agent = ""       # currently active agent (shown in the status bar)
        self._agent_tokens: dict[str, int] = {}   # each agent's context usage
        self._history: list[str] = []  # input history (left/right keys to browse)
        self._hist_idx = 0
        self._draft = ""               # unsent draft (stashed while browsing history, restored at the bottom)
        self._loop: asyncio.AbstractEventLoop | None = None

        # confirmation flow (worker blocks waiting for the UI's answer)
        self._confirm_holder: dict | None = None

        # widgets (output area focusable → supports selection/copy)
        self.output = TextArea(
            text="", read_only=True, scrollbar=True, focusable=True,
            wrap_lines=True, lexer=_OutputLexer(),
        )
        self.input = TextArea(
            # GROWS with the text: height = the message's own wrapped-row count, clamped 2..6 (then scrolls).
            # height is a callable (_input_dim) returning an EXACT dimension, so the box sizes to its CONTENT
            # and never grabs spare layout space (a min/max range let it balloon to ~6 rows on a tall screen).
            # multiline=True is required for the box to grow (multiline=False locks height to 1) — so Enter is
            # rebound to SEND (see _bindings), since a multiline buffer would otherwise insert a newline.
            height=self._input_dim, multiline=True, wrap_lines=True,
            prompt=self._input_prompt, accept_handler=self._on_accept,
        )
        status = Window(
            height=1, content=FormattedTextControl(self._status_fragments),
            style="class:status",
        )
        root = HSplit([
            self.output,                       # combined output area (takes remaining space)
            Window(height=1, content=FormattedTextControl(lambda: [("", "")])),  # blank line above the status bar
            status,                            # status bar
            self.input,                        # input area
        ])
        self.app: Application = Application(
            layout=Layout(root, focused_element=self.input),
            key_bindings=self._bindings(),
            style=_STYLE,
            full_screen=True,
            mouse_support=False,   # hand the mouse back to the terminal → native select/right-click/copy/paste across the window
        )
        # wire backend events / confirmation to the UI
        self.backend.event_sink = self._sink
        self.backend.confirm_fn = self._confirm

    # ---- output ---------------------------------------------------------
    def _input_dim(self) -> Dimension:
        """Input height = the message's wrapped-row count, clamped to 2..6 rows (then it scrolls). EXACT, so
        the box sizes to its CONTENT and never grabs spare layout space (the output area absorbs the rest)."""
        import shutil
        ta = getattr(self, "input", None)
        text = ta.buffer.text if ta is not None else ""
        cols = shutil.get_terminal_size((80, 24)).columns
        w = max(10, cols - 8)                              # rough allowance for the prompt + scrollbar
        rows = sum(max(1, -(-len(ln) // w)) for ln in (text.split("\n") or [""]))   # ceil-div per logical line
        return Dimension.exact(min(6, max(1, rows)))

    @staticmethod
    def _at_bottom(buf) -> bool:
        """The view follows the cursor; cursor at (or 1 char from) the end == the bottom is visible."""
        return buf.cursor_position >= len(buf.text) - 1

    def _jump_end(self) -> None:
        """Jump to the bottom and RESUME following (Ctrl-G / /b / sending a message)."""
        self._follow = True
        buf = self.output.buffer
        buf.set_document(Document(buf.text, cursor_position=len(buf.text)), bypass_readonly=True)

    def _append(self, text: str) -> None:
        self._finalize_stream()        # a whole-line output → end any in-progress stream line
        buf = self.output.buffer
        new = (buf.text + ("\n" if buf.text else "") + text)
        # while following, FORCE cursor to the end (can't drift off under fast appends); else keep position
        pos = len(new) if self._follow else min(buf.cursor_position, len(new))
        buf.set_document(Document(new, cursor_position=pos), bypass_readonly=True)

    def _raw_append(self, text: str) -> None:
        """Append text directly to the buffer end (no auto newline; for in-place stream growth)."""
        buf = self.output.buffer
        new = buf.text + text
        pos = len(new) if self._follow else min(buf.cursor_position, len(new))
        buf.set_document(Document(new, cursor_position=pos), bypass_readonly=True)

    def _append_stream(self, name: str, text: str) -> None:
        """Stream fragment: grow the current agent's stream line in place; on a new agent start a new line with a prefix."""
        if self._streaming != name:
            self._finalize_stream()
            self._streaming = name
            if name == "main":
                prefix = "main ◀ "      # contains ◀ → lexer colors it as the main-agent reply
            else:
                prefix = f"{'    ' * self._depth(name)}[{name}] "
            self._raw_append(("\n" if self.output.buffer.text else "") + prefix)
        self._raw_append(text)

    def _finalize_stream(self) -> None:
        """End the current stream line (subsequent output starts on a new line)."""
        self._streaming = None

    def _safe_call(self, fn, *args) -> None:
        """Safely schedule onto the UI event loop from any thread."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(fn, *args)

    def _sink(self, name: str, kind: str, data: Any) -> None:
        """Agent event (possibly from a background thread) → enqueue."""
        if kind in ("tool_call", "spawn", "assistant", "stream"):
            self._current_agent = name
        if kind == "usage":
            self._agent_tokens[name] = data or 0
            return                       # usage isn't printed, only updates the status bar
        if kind == "stream":
            if data:
                if name == "main":
                    self._streamed_main = True   # set on the worker thread; read by _worker without a race
                self._safe_call(self._q.put_nowait, ("delta", name, data))
            return
        line = self._fmt(name, kind, data)
        if line is not None:
            try:
                self._safe_call(self._q.put_nowait, ("line", line))
            except Exception:
                pass

    @staticmethod
    def _depth(name: str) -> int:
        # main=0; sub_1=1; sub_1_1=2 … (counted by the number of underscores)
        return 0 if name == "main" else name.count("_")

    @classmethod
    def _fmt(cls, name: str, kind: str, data: Any) -> str | None:
        ind = "    " * cls._depth(name)
        tag = "" if name == "main" else f"[{name}] "
        if kind == "tool_call":
            nm = getattr(data, "name", "?")
            args = getattr(data, "arguments", {}) or {}
            # take the "primary argument" as the summary (else spawn_subagent shows as max_tokens=-1); full, not truncated
            key = _PRIMARY_ARG.get(nm)
            if key and key in args:
                brief = str(args[key])
            else:
                brief = next((str(v) for v in args.values()), "")
            brief = " ".join(brief.split())   # flatten newlines/runs → ONE line (multi-line args broke the layout)
            return f"{ind}· {tag}{nm} {brief}".rstrip()
        if kind == "tool_result":
            call, res = data
            body = (getattr(res, "text", "") or "").strip()
            if not body:
                return None
            # edit diff: show the whole thing (so the lexer colors +green/-red)
            if getattr(call, "name", "") in ("edit_file", "write_file") and "@@" in body:
                return f"{ind}{tag}{body}"
            # persistent shell: show the FULL output (multi-line) so the interaction is visible live
            if getattr(call, "name", "") in _FULL_RESULT_TOOLS:
                return f"{ind}{tag}{body}"
            return f"{ind}  ↳ {body.splitlines()[0][:70]}"
        if kind == "spawn":
            d = cls._depth(getattr(data, "id", ""))
            return f"{'    ' * max(d - 1, 0)}⤷ spawn {getattr(data, 'id', '?')}: {getattr(data, 'role', '')}"
        if kind == "done":
            return f"{ind}  ok {getattr(data, 'id', name)} {getattr(data, 'status', 'done')}"
        if kind == "cancelled":
            return f"{ind}{i18n.t('interrupted')}"
        if kind == "compact":
            return f"{ind}· (compacting context…)"
        return None  # final/reasoning/limit/rules not shown directly

    # ---- status bar -----------------------------------------------------
    def _marquee(self, name: str, width: int) -> str:
        """name as-is if it fits width; otherwise scroll it as a marquee (advances with self.frame)."""
        if len(name) <= width:
            return name
        s = name + " · "                          # gap so the wrap-around reads cleanly
        off = (self.frame // 3) % len(s)          # ~0.36s/char at the 0.12s spin tick
        return (s + s)[off:off + width]

    def _status_fragments(self):
        import shutil
        cols = shutil.get_terminal_size((80, 24)).columns
        sl = self.backend.status_left()
        pre: list[tuple[str, str]] = [("class:status", " " + sl)]
        suf: list[tuple[str, str]] = []
        # [DEV] — its own field. Its colored block stays " [DEV] " (trailing space included); the NEXT field
        # drops the leading space of its " │ " separator so there's ONE space before the pipe, not two.
        dev = getattr(self.backend, "dev_mode", False)
        if dev:
            suf.append(("class:devmark", " [DEV] "))
        sep = "│ " if dev else " │ "
        if not getattr(self.backend, "server_online", True):
            suf.append(("class:ctxwarn", f"{sep}{i18n.t('server_down')}"))
            sep = " │ "
        # agent — its OWN field, right after the model name. Width is RESERVED from the configured max sub-
        # depth (worst-case name e.g. "sub_99_99" at max_depth=2), so switching agents never re-lays-out the
        # bar; a longer name marquee-scrolls. Always shown (idle → "main") so it doesn't appear/disappear.
        agent = self._current_agent if (self.busy and self._current_agent) else "main"
        agent_w = len("sub" + "_99" * max(1, self.backend.cfg.max_depth))
        suf.append(("class:status", f"{sep}{self._marquee(agent, agent_w).rjust(agent_w, '-')}"))
        # ctx — that agent's context usage (digits can roll as tokens grow; a rare, event-level reflow)
        n_ctx = (self.backend.info.n_ctx or 0) if self.backend.info else 0
        used = self._agent_tokens.get(agent, 0)
        if n_ctx:
            near = used >= 0.9 * n_ctx * self.backend.cfg.compact_threshold
            suf.append(("class:ctxwarn" if near else "class:status",
                        f" │ {i18n.t('ctx_label')} {used // 1000}k/{n_ctx // 1000}k"))
            if near:
                suf.append(("class:ctxwarn", f" {i18n.t('ctx_warn')}"))
        # activity — its OWN fixed-width field: idle → "Idle", busy → spinner + Working/Outputting. Padded to
        # the widest of the three labels so flipping state never changes the bar's width.
        act_inner = max(len(i18n.t("working")) + 2, len(i18n.t("streaming")) + 2, len(i18n.t("standby")))
        if self.busy:
            spin = _SPIN[self.frame % len(_SPIN)]
            label, cls = ((i18n.t("streaming"), "class:stream") if self._streaming
                          else (i18n.t("working"), "class:working"))
            content = f"{spin} {label}"
        else:
            content, cls = i18n.t("standby"), "class:status"
        gap = max(0, act_inner - len(content))            # right-align the label, dash-filling the slack on the LEFT
        suf.append(("class:status", " │ "))
        suf.append((cls, "-" * gap + content))            # color fills the WHOLE field (dashes + label), no white gap
        # persistent shell sessions — a YELLOW "CLI:n" field, shown ONLY while ≥1 session is open (it lives
        # one turn, so it appears during a turn that opened one and clears when the turn ends).
        shells = self.backend.open_shells() if hasattr(self.backend, "open_shells") else 0
        if shells:
            suf.append(("class:status", " │ "))
            suf.append(("class:cli", f" CLI:{shells} "))
        # elapsed — its OWN field, stopwatch style: 4 chars, dash-padded ("----" at 0 / never-run, "--42",
        # "-156", "9999" capped). Counts while busy, frozen at the last turn's duration while idle. The
        # trailing " │" closes the field (so it's bracketed like the others, not left open before the pad).
        el = min(int(time.time() - self.busy_start) if self.busy else self._elapsed_frozen, 9999)
        el_str = (str(el) if el > 0 else "").rjust(4, "-")
        suf.append(("class:status", f" │ {el_str} │"))
        # esc-to-interrupt hint sits at the far right before /help, ONLY while busy — but its width is
        # RESERVED when idle (blank) so the model field doesn't shift on busy↔idle.
        esc_hint = f"({i18n.t('esc_interrupt')}) "
        right = (esc_hint if self.busy else " " * len(esc_hint)) + "/help "
        # The model-name field takes whatever width is LEFT after every other field is laid out at full size
        # (nothing else is ever truncated), recomputed live from the ACTUAL right cluster. Every other field
        # is fixed-width (agent reserved, activity padded, elapsed 4-char, esc reserved), so the model marquee
        # only re-lays-out on real events (ctx digit roll, terminal resize) — never per-second or per-toggle.
        # widths use the terminal DISPLAY width (get_cwidth: CJK/wide chars count as 2), not len(), so a
        # wide-char session/workdir/model name can't make the line overflow and clip the right side (/help).
        # cols - 1 reserves the last column (some terminals drop the bottom-right cell).
        cw = get_cwidth
        usable = cols - 1
        suf_len = sum(cw(t) for _, t in suf)
        avail = max(8, usable - cw(" " + sl) - cw(" │ ") - suf_len - cw(right) - 1)   # -1 = min pad before right
        name = self.backend.model_short()
        if cw(name) <= avail:
            model_field, self._model_marquee = name, False     # fits → full name, no marquee
        else:
            model_field, self._model_marquee = self._marquee(name, avail), True   # too long → scroll in the leftover
        left = pre + [("class:status", f" │ {model_field}")] + suf
        pad = max(1, usable - (sum(cw(t) for _, t in left) + cw(right)))
        return left + [("class:status", " " * pad + right)]

    # ---- key bindings ---------------------------------------------------
    def _bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-d")
        def _(event):
            event.app.exit()

        out_focus = has_focus(self.output)

        # Tab: toggle focus between input / output area (focus the output to scroll it with the arrows).
        # Focusing the output = entering read/scroll mode → STOP auto-following the newest, otherwise a live
        # token stream keeps yanking the view back to the bottom and you can't scroll up. Ctrl-G resumes.
        @kb.add("tab")
        def _(event):
            lay = event.app.layout
            if lay.has_focus(self.output):
                lay.focus(self.input)
            else:
                self._follow = False
                lay.focus(self.output)

        # output focus is only for keyboard scrolling; Enter or Escape returns to the input.
        # (copy/paste is the terminal's OWN native selection — mouse_support is off, so just select with the mouse.)
        @kb.add("enter", filter=out_focus)
        @kb.add("escape", filter=out_focus)
        def _(event):
            event.app.layout.focus(self.input)

        # input is multiline (so the box can GROW with a long message) → Enter would insert a newline; rebind
        # it (eager, so it wins over the buffer's newline) to SEND. Mirrors the accept_handler: keep text if
        # _submit returns True (e.g. busy), else clear.
        @kb.add("enter", filter=has_focus(self.input), eager=True)
        def _(event):
            if not self._submit(self.input.text):
                self.input.buffer.reset()

        # input focused + not eager: avoid swallowing arrow keys; a lone ESC still triggers interrupt
        @kb.add("escape", filter=has_focus(self.input))
        def _(event):
            if self.busy:
                self.backend.request_cancel()
                self._append(i18n.t("esc_interrupt"))

        # Up/Down = scroll the output area (the mouse wheel sends up/down in the terminal too).
        # Scroll-up pauses auto-follow; scrolling back to the bottom resumes it.
        @kb.add("up")
        def _(event):
            self._follow = False
            self.output.buffer.cursor_up(2)

        @kb.add("down")
        def _(event):
            buf = self.output.buffer
            buf.cursor_down(2)
            if self._at_bottom(buf):
                self._follow = True

        # Left/Right = input history (only at line start/end; otherwise move the cursor normally)
        @kb.add("left", filter=has_focus(self.input))
        def _(event):
            b = event.current_buffer
            if b.cursor_position == 0:
                self._history_nav(-1)
            else:
                b.cursor_left()

        @kb.add("right", filter=has_focus(self.input))
        def _(event):
            b = event.current_buffer
            if b.cursor_position >= len(b.text):
                self._history_nav(+1)
            else:
                b.cursor_right()

        # output area scrolling (full page)
        @kb.add("pageup")
        def _(event):
            self._follow = False
            self.output.buffer.cursor_up(15)

        @kb.add("pagedown")
        def _(event):
            buf = self.output.buffer
            buf.cursor_down(15)
            if self._at_bottom(buf):
                self._follow = True

        @kb.add("c-g")                    # Ctrl-G: jump to latest + resume following
        def _(event):
            self._jump_end()

        return kb

    # ---- input history (left/right keys at line start/end) --------------
    def _history_nav(self, delta: int) -> None:
        if not self._history:
            return
        # before leaving the "draft position", stash the half-typed text
        if self._hist_idx >= len(self._history):
            self._draft = self.input.buffer.text
        new_idx = max(0, min(len(self._history), self._hist_idx + delta))
        if new_idx == self._hist_idx:
            return                      # already at the boundary, don't move
        self._hist_idx = new_idx
        # at the bottom (idx==len) restore the draft, otherwise show that history entry
        text = self._history[new_idx] if new_idx < len(self._history) else self._draft
        self.input.buffer.set_document(Document(text, cursor_position=len(text)))

    def _push_history(self, text: str) -> None:
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._hist_idx = len(self._history)
        self._draft = ""

    # ---- input prompt (becomes y/n while confirming, back to 'you' after) ----
    def _input_prompt(self):
        if self._confirm_holder is not None:
            return [("class:confirmprompt", " y/n ▷ ")]
        return [("class:userprompt", i18n.t("prompt"))]

    # ---- submit ---------------------------------------------------------
    def _on_accept(self, buff) -> bool:
        # return True = keep the input text (used when not sending while busy), False = clear
        return self._submit(buff.text)

    def _submit(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        # waiting for a safety confirmation: treat input as y/n (the input prompt reverts to 'you' automatically)
        if self._confirm_holder is not None:
            ans = text.lower() in ("y", "yes")
            self._append(("✔ " if ans else "✘ ") + text)
            holder, self._confirm_holder = self._confirm_holder, None
            holder["result"] = ans
            holder["ev"].set()
            return False
        self._push_history(text)
        if text.startswith("/"):
            self._jump_end()           # sending → snap to bottom + resume follow
            self._append(i18n.t("prompt") + text)
            cmd0 = text.split()[0]
            # /b: TUI-only — jump to latest & resume auto-scroll (the _jump_end above already did it)
            if cmd0 == "/b":
                return False
            # /exit: first ask whether to update the summary; if yes show "working" then quit
            if cmd0 in ("/exit", "/quit"):
                threading.Thread(target=self._exit_worker, daemon=True).start()
                return False
            # slow commands that call the model (/compact) → run in the background, avoid freezing the UI, clear input immediately
            if cmd0 in ("/compact",):
                threading.Thread(target=self._cmd_worker, args=(text,), daemon=True).start()
                return False
            if self.backend.command(text, self._append):
                self.app.exit()
            return False
        # the model is still replying: don't send, keep the input text, show a busy note
        if self.busy:
            self._append(i18n.t("busy_note"))
            return True
        # send a message → run on a background thread (snap to bottom so you see it + the reply)
        self._jump_end()
        self._append(i18n.t("prompt") + text)
        threading.Thread(target=self._worker, args=(text,), daemon=True).start()
        return False

    def _stop_busy(self) -> None:
        """End a busy period: freeze this turn's elapsed seconds (so the idle bar shows how long it took)."""
        self._elapsed_frozen = min(int(time.time() - self.busy_start), 9999)
        self.busy = False

    def _cmd_worker(self, text: str) -> None:
        self.busy = True
        self.busy_start = time.time()
        try:
            self.backend.command(text, self._sink_line)
        finally:
            self._stop_busy()
            self._invalidate()

    def _exit_worker(self) -> None:
        # ask whether to update the summary; if yes show a "working" spinner then quit
        if self._confirm(None, None, i18n.t("ask_summary")):
            self.busy = True
            self.busy_start = time.time()
            try:
                self.backend._save_progress(self._sink_line)
            finally:
                self.busy = False
                self._invalidate()
        self._safe_call(self.app.exit)

    # non-streamed special returns (error/interrupt/offline/degenerate/iter-limit) → still need printing
    _SPECIAL_PREFIXES = ("[error]", "[interrupted]", "[reached", "[generation aborted: degenerate]", "[server")

    def _worker(self, text: str) -> None:
        self.busy = True
        self.busy_start = time.time()
        self._streamed_main = False     # reset each turn: used to decide whether to re-print at the end
        try:
            result = self.backend.run_message(text)
            # the main agent's real content was already streamed live → don't re-print at the end;
            # but special returns (error/interrupt/degenerate…) were not streamed, so still print them.
            special = str(result).startswith(self._SPECIAL_PREFIXES)
            if special or not self._streamed_main:
                self._sink_line(i18n.t("agent_reply", out=result))
        except Exception as e:  # noqa: BLE001
            self._sink_line(f"[error] {e}")
        finally:
            self._stop_busy()
            self._invalidate()

    def _sink_line(self, line: str) -> None:
        try:
            self._safe_call(self._q.put_nowait, ("line", line))
        except Exception:
            pass

    def _invalidate(self) -> None:
        try:
            self._safe_call(self.app.invalidate)
        except Exception:
            pass

    # ---- confirmation (called from the worker thread, blocks waiting for the UI) ----
    def _confirm(self, tool, args, reason) -> bool:
        ev = threading.Event()
        holder = {"ev": ev, "result": False}
        self._confirm_holder = holder
        self._sink_line(f"!  {reason}  — {i18n.t('confirm_hint')}")
        ev.wait()                       # block the worker until _submit receives y/n
        return bool(holder["result"])

    # ---- background tasks: drain queue + spinner ------------------------
    async def _drain(self) -> None:
        while True:
            first = await self._q.get()
            items = [first]
            while not self._q.empty():     # drain everything currently queued (merge stream fragments in batch)
                items.append(self._q.get_nowait())
            buf: list[str] = []            # buffer consecutive same-agent stream fragments, paste once (avoid per-char rebuilds)
            buf_name: str | None = None

            # takes buf/name as PARAMETERS (not a closure over the loop vars) → safe, and returns the reset buffer
            def flush(b: list[str], nm: str | None) -> list[str]:
                if b and nm is not None:
                    self._append_stream(nm, "".join(b))
                return []

            for it in items:
                if it[0] == "delta":
                    _, name, text = it
                    if name != buf_name:
                        buf = flush(buf, buf_name)
                    buf_name = name
                    buf.append(text)
                else:                       # ("line", text): finalize stream first, then print the whole line
                    buf = flush(buf, buf_name)
                    buf_name = None
                    self._finalize_stream()
                    self._append(it[1])
            flush(buf, buf_name)
            self.app.invalidate()
            # throttle: cap repaints to ~20fps so a fast token stream coalesces into fewer (and bigger)
            # batches instead of rebuilding the whole buffer per token — keeps a big/long stream smooth.
            await asyncio.sleep(0.05)

    async def _spin(self) -> None:
        while True:
            await asyncio.sleep(0.12)
            # tick while busy (spinner) OR the model name is being marqueed (compressed) OR the server is
            # offline (so the OFFLINE indicator appears within a tick, not only on the next user action).
            if self.busy or getattr(self, "_model_marquee", False) \
                    or not getattr(self.backend, "server_online", True):
                self.frame += 1
                self.app.invalidate()

    def _startup_banner(self) -> None:
        """On startup, show in the output area: session/workdir/model + falamus.md load status (visible)."""
        from pathlib import Path
        b = self.backend
        model = (b.info.name or "?").split("/")[-1] if b.info else "?"
        self._append(f"falamus · session {b.runtime.session.sid} · {b.workdir} · {model}")
        p = Path(b.workdir) / "falamus.md"
        # Plain-chat mode (model has no tools, OR no usable shell): tools are off, but an EXISTING
        # falamus.md is STILL injected as rules — plain chat only skips CREATING one. Report accurately.
        plain = bool(b.force_plain_chat or (b.info and not b.info.supports_tools))
        if plain:
            if p.exists():
                n = len(p.read_text(encoding="utf-8", errors="replace"))
                self._append(i18n.t("no_tools_chat_rules", n=n))
            else:
                self._append(i18n.t("no_tools_chat"))
        elif p.exists():
            n = len(p.read_text(encoding="utf-8", errors="replace"))
            self._append(i18n.t("helper_loaded", n=n))
        self._append(i18n.t("welcome"))

    def run(self) -> None:
        self._startup_banner()

        async def _amain() -> None:
            self._loop = asyncio.get_running_loop()
            drain = asyncio.create_task(self._drain())
            spin = asyncio.create_task(self._spin())
            try:
                await self.app.run_async()
            finally:
                drain.cancel()
                spin.cancel()

        asyncio.run(_amain())
