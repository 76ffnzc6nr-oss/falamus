"""Agent main loop.

A generic agent: given a client, a tool registry, and a system prompt, runs the loop
"chat → (model wants a tool?) → execute → feed back → continue" until the model stops
asking for tools.

Tool-calling requires native function calling. A model WITHOUT tool support is not given any
tools — it degrades to plain chat (one reply, no tool-call parsing).

This Agent is the shared base for both the main agent and sub-agents (they differ only in
system prompt, allowed_tools, and whether spawn_subagent is attached).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from typing import Any

from falamus.tools.registry import ToolRegistry, ToolResult

from .client import ChatResponse, DegenerateOutput, LLMClient, StreamInterrupted, ToolCall

# returned with this prefix when degenerate generation (repetition spiral) is aborted;
# subagent treats it as a failure and reports it up
_DEGEN_TAG = "[generation aborted: degenerate]"

# repeat guard, sub-agent abort prefixes: a sub-agent that keeps re-issuing the SAME call is a dead end, so
# we abort it and hand an HONEST status to the parent — distinguishing whether that repeated call had
# actually SUCCEEDED (work already produced; treat the step as DONE) from never succeeding (step FAILED).
# (Without this, a sub that loops on an already-successful write escapes the soft nudge by writing a stub
# and then falsely reports success — observed in test4.)
_STOP_REPEAT_DONE = "[stopped: repeated an already-SUCCESSFUL call"
_STOP_REPEAT_FAIL = "[stopped: repeated a FAILING call"

# repeat guard: an EXACT-identical tool call (same name + args) made this many times in a row is a stuck
# loop with no progress. Sub-agents abort at _REPEAT_BLOCK (and report up). The main agent (unlimited, no
# parent) escalates: block with guidance first, hard-stop only at _REPEAT_ABORT.
_REPEAT_BLOCK = 3
_REPEAT_ABORT = 6

# UI / log event callback: on_event(kind, data)
#   kind ∈ {"assistant", "tool_call", "tool_result", "final", "limit"}
EventCb = Callable[[str, Any], None]


class Agent:
    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        *,
        system_prompt: str = "",
        allowed_tools: list[str] | None = None,
        name: str = "agent",
        max_iters: int = 20,
        enable_thinking: Any = "defer",  # "defer"=follow the client (decided by config); True/False/None=explicit
        max_tokens: int | None = None,   # None=use client default; -1=unbounded (until EOS)
        on_event: EventCb | None = None,
        context_manager: Any = None,
        checkpoint_cb: Callable[[str, list[dict[str, Any]]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        turn_reminder: str = "",          # background summary re-sent each turn (keeps self-awareness)
        error_log: Any = None,            # centralized error log (ErrorLog)
        plain_chat: bool = False,         # force the tool-less plain-chat path even if the model supports tools
    ) -> None:
        self.client = client
        self.registry = registry
        self.allowed_tools = allowed_tools
        self.plain_chat = plain_chat
        self.name = name
        self.max_iters = max_iters
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.on_event = on_event
        self.cm = context_manager
        self.checkpoint_cb = checkpoint_cb
        self.cancel_check = cancel_check
        self.turn_reminder = turn_reminder
        self.error_log = error_log
        # All STANDING system content (base prompt + falamus.md, compaction summary, the per-turn brief)
        # lives inside the ONE leading system message — never as separate system messages mid-conversation
        # (strict chat templates, e.g. Qwen, reject those). _compose_system rebuilds messages[0] before every
        # call. NOTE: the memo is deliberately NOT folded in here — it's an external, tool-accessed store
        # (see the `memo` tool), so it never mutates the prompt prefix and the KV cache stays valid.
        self._base_system = system_prompt   # immutable base (prompt + falamus.md); folded-in pieces are separate
        self._summary = ""                  # accumulated compaction summary (absorbed from the context manager)
        self._last_sig: str | None = None   # last tool call signature (name+args) — for the repeat guard
        self._repeat = 0                    # consecutive identical tool calls
        self._last_ok: bool | None = None   # did the most recently EXECUTED tool call succeed? (for the guard's report)
        self._aborted = False               # set when the repeat guard escalates to stop the agent
        self._abort_reason = ""             # the honest stop message handed back to the parent on abort
        self.is_sub = name != "main"        # sub-agents abort-and-report on a repeat loop; main escalates
        self.messages: list[dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def restore(self, messages: list[dict[str, Any]]) -> None:
        """Replace the current content with a restored message history (for resume).

        The leading system message holds base + folded-in summary/brief; split them back out so the
        agent's fields are correct and _compose_system can rebuild it cleanly. (The memo is external —
        nothing to recover from the prompt.)
        """
        self.messages = list(messages)
        if self.messages and self.messages[0].get("role") == "system" \
                and isinstance(self.messages[0].get("content"), str):
            base, summary = self._split_system(self.messages[0]["content"])
            self._base_system, self._summary = base, summary

    def _checkpoint(self) -> None:
        if self.checkpoint_cb:
            self.checkpoint_cb(self.name, self.messages)

    # ---- public ---------------------------------------------------------
    _BRIEF_TAG = "[brief]"
    _SUMMARY_TAG = "[Summary of earlier conversation]"   # must match ContextManager's note

    def _absorb_summary(self) -> None:
        """Pull any standalone [Summary] system note (produced by the context manager during compaction)
        out of the message list and into self._summary, so it ends up inside the ONE leading system msg."""
        kept: list[dict[str, Any]] = []
        for i, m in enumerate(self.messages):
            if (i != 0 and m.get("role") == "system" and isinstance(m.get("content"), str)
                    and m["content"].startswith(self._SUMMARY_TAG)):
                text = m["content"][len(self._SUMMARY_TAG):].strip()
                self._summary = f"{self._summary}\n\n{text}".strip() if self._summary else text
            else:
                kept.append(m)
        self.messages = kept

    def _compose_system(self) -> None:
        """Rebuild the single leading system message = base + summary + brief. Keeps ALL standing
        instructions in the one place strict templates require (the start), refreshed before each call.
        (The memo is NOT here — it lives in the external `memo` store, accessed via the tool.)"""
        if not (self.messages and self.messages[0].get("role") == "system"):
            return
        parts = [self._base_system]
        if self._summary:
            parts.append(f"{self._SUMMARY_TAG}\n{self._summary}")
        if self.turn_reminder:
            parts.append(f"{self._BRIEF_TAG} {self.turn_reminder}")
        self.messages[0]["content"] = "\n\n".join(parts)

    def _split_system(self, content: str) -> tuple[str, str]:
        """Inverse of _compose_system: recover (base, summary) from a composed leading system msg."""
        def section(tag: str, nxt: list[str]) -> str:
            i = content.find("\n\n" + tag)
            if i == -1:
                return ""
            start = i + len("\n\n" + tag)
            end = len(content)
            for t in nxt:
                j = content.find("\n\n" + t, start)
                if j != -1:
                    end = min(end, j)
            return content[start:end].strip()
        firsts = [content.find("\n\n" + t) for t in (self._SUMMARY_TAG, self._BRIEF_TAG)]
        firsts = [p for p in firsts if p != -1]
        base = (content[:min(firsts)] if firsts else content).strip()
        summary = section(self._SUMMARY_TAG, [self._BRIEF_TAG])
        return base, summary

    def run(self, user_input: str) -> str:
        """Feed one user input, run the full loop, return the final text reply."""
        # brief/summary are folded into the leading system message by _compose_system (called in _chat),
        # not appended as separate mid-conversation system messages.
        self.messages.append({"role": "user", "content": user_input})
        if self.client.info is None:
            self.client.detect()
        native = bool(self.client.info and self.client.info.supports_tools) and not self.plain_chat
        return self._loop_native() if native else self._chat_only()

    # ---- events ---------------------------------------------------------
    def _emit(self, kind: str, data: Any) -> None:
        if self.on_event:
            self.on_event(kind, data)

    def _cancelled(self) -> bool:
        return bool(self.cancel_check and self.cancel_check())

    def _err(self, kind: str, message: str, detail: str = "") -> None:
        if self.error_log:
            try:
                self.error_log.log(self.name, kind, message, detail)
            except Exception:
                pass

    # ---- wrapper: compact → chat → calibrate ----------------------------
    def _chat(self, tools: list[dict[str, Any]] | None) -> ChatResponse:
        # keep ALL standing system content in the ONE leading system message (strict-template safe).
        self._compose_system()           # fold current brief/summary in (so compaction counts them too)
        if self.cm is not None:
            self.messages = self.cm.maybe_compact(self.messages)
        self._absorb_summary()           # pull any new [Summary] note compaction added into self._summary
        self._compose_system()           # re-fold so the new summary lands inside the leading system msg
        # guard_degenerate: switch to streaming, watch for the "repetition spiral" (after 2 min with a
        # consecutively-repeating tail → abort). on_delta emits each content chunk as a "stream" event for live UI.
        on_delta = lambda d: self._emit("stream", d)  # noqa: E731
        # cancel_check lets ESC / server-offline interrupt mid-generation (per streamed chunk)
        if self.enable_thinking == "defer":   # follow the client setting (decided by config.thinking)
            resp = self.client.chat(self.messages, tools=tools, max_tokens=self.max_tokens,
                                    guard_degenerate=True, on_delta=on_delta, cancel_check=self.cancel_check)
        else:
            resp = self.client.chat(
                self.messages, tools=tools, enable_thinking=self.enable_thinking,
                max_tokens=self.max_tokens, guard_degenerate=True, on_delta=on_delta,
                cancel_check=self.cancel_check,
            )
        if self.cm is not None:
            self.cm.observe(resp)
            self._emit("usage", self.cm._last_server_tokens)   # for the status bar to show this agent's usage
        return resp

    def _iter_budget(self):
        """Iteration budget for a turn. max_iters <= 0 means UNLIMITED — used for the main agent, whose
        real death-loop guard is the consecutive-sub-failure circuit breaker, not an iteration cap.
        Sub-agents keep a finite cap (their backstop against single-unit loops, e.g. repeating ls)."""
        return itertools.count() if self.max_iters <= 0 else range(self.max_iters)

    # ---- native function-calling loop -----------------------------------
    def _loop_native(self) -> str:
        tools = self.registry.to_openai_tools(self.allowed_tools)
        for _ in self._iter_budget():
            if self._cancelled():
                self._emit("cancelled", None)
                return "[interrupted]"
            try:
                resp = self._chat(tools)
            except StreamInterrupted as e:
                # the partial stream is NOT added to the conversation (would pollute context on resume);
                # save its tail to the error log so an interrupted/degenerate generation is still diagnosable.
                if e.partial.strip():
                    self._err("interrupted_partial", f"{len(e.partial)} chars discarded", e.partial[-1500:])
                self._emit("cancelled", None)
                return "[interrupted]"
            except DegenerateOutput as e:
                self._err("degenerate", str(e))
                self._emit("cancelled", None)
                return f"{_DEGEN_TAG} {e}"
            if resp.reasoning:
                self._emit("assistant", {"reasoning": resp.reasoning})
            if resp.wants_tool:
                self.messages.append(self._assistant_tool_msg(resp))
                for call in resp.tool_calls:
                    self._run_tool(call)
                self._checkpoint()
                if self._aborted:
                    return self._abort_reason or "[stopped: repeated identical tool call with no progress]"
                continue
            self.messages.append({"role": "assistant", "content": resp.content})
            self._checkpoint()
            self._emit("final", resp.content)
            return resp.content
        self._emit("limit", self.max_iters)

        self._err("max_iters", f"reached max_iters={self.max_iters} (possible loop)")
        return "[reached the max tool-iteration limit, stopping]"

    def _repeat_guard(self, call: ToolCall):
        """Track consecutive EXACT-identical tool calls; return a block ToolResult once a true loop is
        detected, else None. Only byte-identical (name+args) repeats count → no false-positives on
        near-identical/legitimate re-checks. Escalates to abort the agent (matters for the unlimited main)."""
        sig = f"{call.name}:{json.dumps(call.arguments or {}, sort_keys=True, ensure_ascii=False)}"
        self._repeat = self._repeat + 1 if sig == self._last_sig else 1
        self._last_sig = sig
        if self._repeat < _REPEAT_BLOCK:
            return None
        self._err("repeat_block", f"{call.name} repeated {self._repeat}x (identical)", "")
        # Sub-agent: a repeated call is a dead end → abort NOW and hand the parent an HONEST status, instead
        # of letting the sub limp on (the soft "change args" nudge lets a weak model escape by writing a stub
        # and then falsely report success). Report whether the repeated call had actually SUCCEEDED.
        if self.is_sub:
            self._aborted = True
            if self._last_ok:
                self._abort_reason = (
                    f"{_STOP_REPEAT_DONE}]: this sub-agent issued {call.name} {self._repeat}× with identical "
                    f"arguments; an earlier identical call had SUCCEEDED, so this step's output is ALREADY "
                    f"produced — it looped instead of finishing. Treat the step as DONE: verify the file(s) "
                    f"in the workspace and continue; do NOT re-dispatch this same step.")
            else:
                self._abort_reason = (
                    f"{_STOP_REPEAT_FAIL}]: this sub-agent issued {call.name} {self._repeat}× with identical "
                    f"arguments and it never succeeded — the step did NOT complete. Re-dispatch it differently "
                    f"(smaller/clearer), or report the gap to the user.")
            return ToolResult.error(self._abort_reason)
        # Main agent (unlimited, no parent to return to): escalate — guidance first, hard-stop at _REPEAT_ABORT.
        if self._repeat >= _REPEAT_ABORT:
            self._aborted = True
            self._abort_reason = (f"Stopping: {call.name} was called with identical arguments {self._repeat} "
                                  f"times and is not making progress.")
            return ToolResult.error(f"[loop] {self._abort_reason}")
        return ToolResult.error(
            f"[loop] You have called {call.name} with identical arguments {self._repeat} times in a row with "
            f"no progress — repeating it will NOT help. STOP repeating: change the arguments, use a different "
            f"tool/approach, or report the problem and move on.")

    def _run_tool(self, call: ToolCall) -> None:
        self._emit("tool_call", call)
        blocked = self._repeat_guard(call)
        if blocked is not None:
            result = blocked                 # guard tripped: do NOT execute, and keep _last_ok as-is
        else:
            result = self.registry.execute(call)
            self._last_ok = not result.is_error   # remember success of the most recent EXECUTED call
        if result.is_error:
            brief = next((str(v) for v in (call.arguments or {}).values()), "")
            self._err("tool_error", f"{call.name}({brief[:80]})", result.text)
        self._emit("tool_result", (call, result))
        self.messages.append(result.to_message(call.id, call.name))

    @staticmethod
    def _assistant_tool_msg(resp: ChatResponse) -> dict[str, Any]:
        """Rebuild the assistant tool_call message in CANONICAL OpenAI form (function.arguments as a
        JSON string). self.messages always stays OpenAI-shaped so the context manager / checkpoints see
        one format; the ollama client rewrites it to native form at the send boundary
        (LLMClient._to_ollama_messages).
        """
        return {
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": c.id,
                    "function": {
                        "name": c.name,
                        "arguments": json.dumps(c.arguments, ensure_ascii=False),
                    },
                }
                for c in resp.tool_calls
            ],
        }

    # ---- plain chat (model without tool support) ------------------------
    def _chat_only(self) -> str:
        """Model without native tool support: plain chat — one reply, NO tools sent, no tool-call parsing."""
        if self._cancelled():
            self._emit("cancelled", None)
            return "[interrupted]"
        try:
            resp = self._chat(None)           # no tools → the model just converses
        except StreamInterrupted:
            self._emit("cancelled", None)
            return "[interrupted]"
        except DegenerateOutput as e:
            self._err("degenerate", str(e))
            self._emit("cancelled", None)
            return f"{_DEGEN_TAG} {e}"
        self.messages.append({"role": "assistant", "content": resp.content})
        self._checkpoint()
        self._emit("final", resp.content)
        return resp.content


if __name__ == "__main__":
    from falamus.tools import default_registry

    cli = LLMClient("http://localhost:8080")
    cli.detect()
    reg = default_registry("/home/c/helper")

    def log(kind: str, data: Any) -> None:
        if kind == "tool_call":
            print(f"  · call {data.name}({data.arguments})")
        elif kind == "tool_result":
            call, res = data
            print(f"  · result {call.name}: {res.text[:80].strip()}…")
        elif kind == "final":
            print(f"  * final: {data}")

    agent = Agent(cli, reg, name="main",
                  system_prompt="You are a local assistant that can use tools. Reply in English.",
                  on_event=log)
    print("=== multi-step task: list the working directory, then read requirements.txt ===")
    out = agent.run("First list the files in the current working directory, then read requirements.txt to me.")
    print("\nreturned:", out)
