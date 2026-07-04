"""Automatic context compaction.

Layered strategy (principle: avoid calling the model when possible):
  1. Rule-based trimming (no LLM): truncate long tool outputs, keep only the most recent
     screenshot, turn the rest into text placeholders.
  2. LLM summary (when needed): condense older conversation into one summary, keep the most
     recent N messages verbatim.

Protected (never touched): system messages (including falamus.md rules) and the most recent
keep_recent messages.
Triggered when: estimated tokens / n_ctx exceeds threshold.

Token estimation uses a light heuristic (CJK ~1 tok/char, otherwise ~4 chars/tok); if the
server reports prompt_tokens it is used to calibrate (observe).
"""

from __future__ import annotations

import json
from typing import Any

_IMG_TOKENS = 1024      # rough cost of one visual image
_TOOL_TRIM = 800        # char cap for a single tool output during rule trimming


def estimate_text_tokens(s: str) -> int:
    if not s:
        return 0
    cjk = sum(1 for c in s if ord(c) > 0x2E00)
    other = len(s) - cjk
    return cjk + other // 4 + 1


def estimate_message_tokens(m: dict[str, Any]) -> int:
    total = 4  # per-message structural overhead
    content = m.get("content")
    if isinstance(content, str):
        total += estimate_text_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                total += estimate_text_tokens(part.get("text", ""))
            elif part.get("type") == "image_url":
                total += _IMG_TOKENS
    for tc in m.get("tool_calls", []) or []:
        args = tc.get("function", {}).get("arguments", "")
        total += estimate_text_tokens(args if isinstance(args, str) else json.dumps(args))
    return total


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


class ContextManager:
    def __init__(
        self,
        client: Any,
        n_ctx: int,
        *,
        threshold: float = 0.7,
        keep_recent: int = 6,
        on_event: Any = None,
    ) -> None:
        self.client = client
        self.n_ctx = n_ctx or 8192
        self.threshold = threshold
        self.keep_recent = keep_recent
        self.on_event = on_event
        self._last_server_tokens = 0

    @property
    def budget(self) -> int:
        return int(self.n_ctx * self.threshold)

    def observe(self, resp: Any) -> None:
        """Record the server-reported actual prompt_tokens (for calibration)."""
        pt = (getattr(resp, "usage", {}) or {}).get("prompt_tokens")
        if pt:
            self._last_server_tokens = pt

    def current_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate current tokens; if a server value exists, take the larger (conservative)."""
        est = estimate_tokens(messages)
        return max(est, self._last_server_tokens)

    # ---- main flow ------------------------------------------------------
    def maybe_compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.current_tokens(messages) < self.budget:
            return messages
        self._emit(f"compaction triggered (~{self.current_tokens(messages)} tok / budget {self.budget})")

        # layer 1: rule-based trimming
        messages = self._rule_trim(messages)
        if estimate_tokens(messages) < self.budget:
            self._emit("within budget after rule trimming")
            self._last_server_tokens = 0
            return messages

        # layer 2: LLM summary of older conversation
        messages = self._llm_summarize(messages)
        self._last_server_tokens = 0
        self._emit(f"~{estimate_tokens(messages)} tok after summary")
        return messages

    def force(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Manual compaction (for /compact): rule trim + LLM summary, ignoring the threshold."""
        messages = self._rule_trim(messages)
        messages = self._llm_summarize(messages)
        self._last_server_tokens = 0   # drop the stale (pre-compact) server count, else the next turn's
        return messages                # current_tokens() stays inflated → a redundant auto-compaction fires

    # ---- split: fixed-protected head (system) and tail (recent N) --------
    def _split(self, messages: list[dict[str, Any]]):
        head = [m for m in messages if m.get("role") == "system"]
        body = [m for m in messages if m.get("role") != "system"]
        if len(body) <= self.keep_recent:
            return head, [], body
        return head, body[: -self.keep_recent], body[-self.keep_recent:]

    # ---- layer 1: rule-based trimming (only touches "old" messages) ------
    def _rule_trim(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        head, old, recent = self._split(messages)
        seen_image = False
        # scan old newest-to-oldest: keep only the most recent image, replace the rest with placeholders; trim long text
        trimmed_rev: list[dict[str, Any]] = []
        for m in reversed(old):
            m = dict(m)
            content = m.get("content")
            if isinstance(content, list):
                new_parts = []
                for part in content:
                    if part.get("type") == "image_url":
                        if seen_image:
                            new_parts.append({"type": "text", "text": "[older screenshot omitted]"})
                        else:
                            seen_image = True
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                m["content"] = new_parts
            elif isinstance(content, str) and len(content) > _TOOL_TRIM and m.get("role") in ("tool", "user"):
                m["content"] = content[: _TOOL_TRIM // 2] + "\n…[trimmed]…\n" + content[-_TOOL_TRIM // 2:]
            trimmed_rev.append(m)
        return head + list(reversed(trimmed_rev)) + recent

    # ---- layer 2: LLM summary ------------------------------------------
    def _llm_summarize(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        head, old, recent = self._split(messages)
        if not old:
            return messages
        transcript = self._render(old)
        prompt = (
            "Condense the following conversation into a concise summary, keeping: what was done, "
            "key conclusions/data, todos or open questions, and important file paths. Use bullet "
            "points in the conversation's main language; no pleasantries.\n\n" + transcript
        )
        try:
            resp = self.client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=1024, enable_thinking=False,
            )
            summary = resp.content or "(summary generation failed)"
        except Exception as e:  # noqa: BLE001
            summary = f"(summary failed, keeping as-is: {e})"
            return messages
        note = {"role": "system", "content": "[Summary of earlier conversation]\n" + summary}
        return head + [note] + recent

    @staticmethod
    def _render(messages: list[dict[str, Any]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "[image]") for p in content if isinstance(p, dict)
                )
            if m.get("tool_calls"):
                names = ", ".join(tc.get("function", {}).get("name", "") for tc in m["tool_calls"])
                content = (content or "") + f" [called tools: {names}]"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _emit(self, msg: str) -> None:
        if self.on_event:
            self.on_event("compact", msg)


if __name__ == "__main__":
    # force a trigger with a low n_ctx to verify trimming and protection logic
    from falamus.core.client import LLMClient

    cli = LLMClient("http://localhost:8080")
    cli.detect()
    cm = ContextManager(cli, n_ctx=400, threshold=0.7, keep_recent=2,
                        on_event=lambda k, d: print(f"  [{k}] {d}"))

    msgs = [{"role": "system", "content": "You are an assistant. (rules)"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"Q{i}: " + "a long string of content. " * 30})
        msgs.append({"role": "assistant", "content": f"A{i}. " * 20})

    print("messages before:", len(msgs), "est tok:", estimate_tokens(msgs))
    out = cm.maybe_compact(msgs)
    print("messages after:", len(out), "est tok:", estimate_tokens(out))
    print("retained system/summary:")
    for m in out:
        if m["role"] == "system":
            print("  -", m["content"][:60].replace("\n", " "))
