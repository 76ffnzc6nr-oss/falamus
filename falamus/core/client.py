"""LLM client layer.

A unified interface to two local backends:
  - llama.cpp  → OpenAI-compatible `/v1/chat/completions`
  - ollama     → native `/api/chat`

Responsibilities:
  1. On startup, auto-detect the backend type, model name, context size, whether it is
     multimodal, and whether it supports tools.
  2. Provide a unified `chat()` that splits the response into reasoning / content / tool_calls
     (some Gemma builds have a reasoning channel that "thinks" before the real output).

This module deliberately depends only on httpx to stay lightweight.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

# After the model reports finish_reason, the proper stream end is still `data: [DONE]` (it carries the
# trailing usage chunk → accurate token count). But if [DONE] never arrives within this grace (a dead /
# kept-alive-idle connection), a watchdog closes the socket so the read aborts instead of hanging forever.
_DONE_GRACE = 15.0


def _keepalive_socket_options() -> list[tuple[int, int, int]]:
    """TCP keepalive so a SILENTLY-dead connection (network drop / NAT idle-eviction / hard crash — no FIN/RST)
    is detected by the OS in ~30s instead of blocking a read=None socket forever. The peer's kernel ACKs
    keepalive probes even while the server app is busy generating, so this does NOT false-abort a slow-but-
    alive request (unlike a blind read timeout). Only options whose constants exist on this platform are set
    (Linux: KEEPIDLE/INTVL/CNT; macOS: TCP_KEEPALIVE for idle; Windows: SO_KEEPALIVE only)."""
    opts: list[tuple[int, int, int]] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    tcp = socket.IPPROTO_TCP
    if hasattr(socket, "TCP_KEEPIDLE"):          # Linux: seconds idle before the first probe
        opts.append((tcp, socket.TCP_KEEPIDLE, 15))
    elif hasattr(socket, "TCP_KEEPALIVE"):       # macOS spelling of the same idle setting
        opts.append((tcp, socket.TCP_KEEPALIVE, 15))
    if hasattr(socket, "TCP_KEEPINTVL"):         # seconds between probes
        opts.append((tcp, socket.TCP_KEEPINTVL, 5))
    if hasattr(socket, "TCP_KEEPCNT"):           # failed probes before the connection is declared dead
        opts.append((tcp, socket.TCP_KEEPCNT, 3))
    return opts

# distinguish "unspecified" from "explicitly None (use server default)"
_UNSET = object()


class DegenerateOutput(RuntimeError):
    """Generation-layer degeneration: the model keeps repeating the same span within a single
    generation (a loop); the generation has been aborted."""


class StreamInterrupted(RuntimeError):
    """The user interrupted (ESC) / the server went offline mid-stream; generation was aborted.
    Carries the partial text streamed so far (for diagnostics — it's never committed to the conversation)."""

    def __init__(self, partial: str = "") -> None:
        super().__init__("stream interrupted")
        self.partial = partial


# ---- degenerate-repetition watchdog --------------------------------------------------------------------
# A runaway loop = the RECENT output adds ~no NEW information given what came just before, at ANY scale
# (a repeated word, sentence, or a whole re-generated block). We measure that "marginal novelty" with
# compression: how many extra compressed bytes the last _DEGEN_PROBE bytes cost given the preceding
# context. Near-zero → the tail is a repeat of earlier output → degenerate. This is scale-invariant (ONE
# rule for small/medium/large loops — no window tiers) and dilution-free (it's the recent slice's novelty,
# not a whole window's absolute ratio, so prior normal text doesn't mask a new loop). _watch additionally
# requires the low-novelty to PERSIST, so a BOUNDED legitimate repeat (says something 3× then moves on)
# isn't flagged — only an UNBOUNDED loop, which never moves on, is.
_DEGEN_CONTEXT = 16384   # how far back a repeat can be matched (~the zlib window)
_DEGEN_PROBE = 256       # the recent slice whose marginal novelty we measure (bytes)
_DEGEN_MIN = 2048        # don't judge until the tail is at least this long (bytes)
_DEGEN_RATIO = 0.08      # marginal novelty below this = redundant (loop ~0.01; a numbered list ~0.11, safe)
_DEGEN_SUSTAIN = 3       # consecutive low-novelty checks (~3s) before calling it degenerate


def _tail_novelty(text: str) -> float:
    """Marginal compressed novelty of the last _DEGEN_PROBE bytes given the preceding context, in [0,1].
    ~0 = the recent output is redundant (a repeat of earlier output, at any scale); ~0.3-0.6 for fresh
    code/prose. 1.0 (= 'not degenerate') until the tail is long enough to judge."""
    data = text[-_DEGEN_CONTEXT:].encode("utf-8", "replace")
    if len(data) < _DEGEN_MIN:
        return 1.0
    probe = min(_DEGEN_PROBE, len(data) // 2)
    whole = len(zlib.compress(data, 6))
    without = len(zlib.compress(data[:-probe], 6))
    return max(0.0, (whole - without) / probe)


# ──────────────────────────────────────────────────────────────────────────
# data structures
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ToolCall:
    """A tool-call request emitted by the model (not yet executed)."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """One chat response, with reasoning channel and real content already separated."""

    content: str = ""               # the real reply shown to the user
    reasoning: str = ""             # reasoning-channel content (kept out of the main window)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ModelInfo:
    """Detected model/backend capabilities."""

    backend: str                    # "llama_cpp" | "ollama"
    name: str
    n_ctx: int = 0
    vision: bool = False
    supports_tools: bool = False
    base_url: str = ""

    def summary(self) -> str:
        bits = [
            f"backend={self.backend}",
            f"model={self.name}",
            f"n_ctx={self.n_ctx}",
            f"vision={'yes' if self.vision else 'no'}",
            f"tools={'yes' if self.supports_tools else 'no'}",
        ]
        return "  ".join(bits)


class BackendError(RuntimeError):
    """Client-layer error (connection, response format, etc.)."""


# ──────────────────────────────────────────────────────────────────────────
# client layer
# ──────────────────────────────────────────────────────────────────────────
class LLMClient:
    """A client supporting both llama.cpp and ollama."""

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        backend: str = "auto",       # "auto" | "llama_cpp" | "ollama" | "anthropic"
        *,
        api_key: str | None = None,  # cloud backends (anthropic): sent as an auth header, never persisted here
        default_max_tokens: int = -1,   # internal fallback when a call passes None; -1 = unbounded (every real caller passes its own cap)
        default_temperature: float = 0.2,
        enable_thinking: bool | None = False,  # thinking off by default: saves tokens, lower latency; None=server default
        timeout: float = 15.0,    # connect cap only; generation itself is unbounded (read=None)
        repeat_penalty: float = 1.15,  # sampler: >1 discourages repeating recent tokens (1.0 = off)
        repeat_last_n: int = 256,      # sampler: how many recent tokens the repeat penalty looks back over
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.backend = backend
        self.api_key = api_key
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.enable_thinking = enable_thinking
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        # Drop the "total generation time" cap (read=None): large files/long code can take as long as needed.
        # Keep the connect cap → an unreachable (offline) server fails fast instead of hanging. TCP keepalive
        # (on the transport) is what rescues a SILENTLY-dead connection: with read=None there is no app-level
        # timeout to false-abort a slow generation, so the OS keepalive — which the live peer keeps ACKing —
        # is the right detector for "the connection actually died" vs "the server is just slow".
        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=float(timeout), read=None, write=None, pool=float(timeout)),
            transport=httpx.HTTPTransport(socket_options=_keepalive_socket_options()),
        )
        self.info: ModelInfo | None = None
        self.error_log: Any = None   # set by Backend.build(); connection events are recorded here

    def _log_conn(self, kind: str, message: str) -> None:
        """Record a connection event (keepalive abort / watchdog close / no-reply) to the error log if one
        is attached. Transport-level, so the agent name is just 'client'. Never raises."""
        if self.error_log is not None:
            try:
                self.error_log.log("client", kind, message)
            except Exception:  # noqa: BLE001
                pass

    def ping(self, timeout: float = 5.0) -> bool:
        """Heartbeat: is the server online (llama.cpp /health, ollama /api/version)."""
        if self.backend == "anthropic":
            return True   # a hosted API has no /health; don't let the heartbeat mark it offline
        for path in ("/health", "/api/version"):
            try:
                if self._http.get(f"{self.base_url}{path}", timeout=timeout).status_code == 200:
                    return True
            except Exception:
                continue
        return False

    # ---- lifecycle ------------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- detection ------------------------------------------------------
    def detect(self) -> ModelInfo:
        """Detect the backend and model capabilities; stored in self.info and returned."""
        backend = self.backend
        if backend == "auto":
            backend = self._detect_backend()

        if backend == "ollama":
            self.info = self._detect_ollama()
        elif backend == "llama_cpp":
            self.info = self._detect_llamacpp()
        elif backend == "anthropic":
            self.info = self._detect_anthropic()
        else:
            raise BackendError(f"Unknown backend: {backend}")

        self.backend = self.info.backend
        if not self.model:
            self.model = self.info.name
        return self.info

    def _detect_backend(self) -> str:
        # ollama has /api/version; llama.cpp doesn't, but has /props or /health
        try:
            r = self._http.get(f"{self.base_url}/api/version", timeout=5.0)
            if r.status_code == 200 and "version" in r.json():
                return "ollama"
        except Exception:
            pass
        try:
            r = self._http.get(f"{self.base_url}/health", timeout=5.0)
            if r.status_code == 200:
                return "llama_cpp"
        except Exception:
            pass
        # last resort: try llama.cpp's /props
        try:
            r = self._http.get(f"{self.base_url}/props", timeout=5.0)
            if r.status_code == 200:
                return "llama_cpp"
        except Exception:
            pass
        raise BackendError(
            f"Could not determine the backend type of {self.base_url} (both ollama / llama.cpp probes failed)"
        )

    def _detect_llamacpp(self) -> ModelInfo:
        name = self.model or ""
        n_ctx = 0
        vision = False
        supports_tools = False
        # prefer /props (most complete info)
        try:
            props = self._http.get(f"{self.base_url}/props", timeout=8.0).json()
            name = props.get("model_alias") or name
            gen = props.get("default_generation_settings", {})
            n_ctx = gen.get("n_ctx") or props.get("n_ctx") or 0
            vision = bool(props.get("modalities", {}).get("vision", False))
            caps = props.get("chat_template_caps", {})
            supports_tools = bool(caps.get("supports_tools", False))
        except Exception:
            pass
        # supplement: /v1/models for name and n_ctx
        if not name or not n_ctx:
            try:
                models = self._http.get(f"{self.base_url}/v1/models", timeout=8.0).json()
                data = (models.get("data") or [{}])[0]
                name = name or data.get("id", "")
                n_ctx = n_ctx or data.get("meta", {}).get("n_ctx", 0)
                caps = (models.get("models") or [{}])[0].get("capabilities", [])
                vision = vision or ("multimodal" in caps)
            except Exception:
                pass
        return ModelInfo(
            backend="llama_cpp",
            name=name,
            n_ctx=n_ctx,
            vision=vision,
            supports_tools=supports_tools,
            base_url=self.base_url,
        )

    def _detect_ollama(self) -> ModelInfo:
        name = self.model or ""
        if not name:
            try:
                tags = self._http.get(f"{self.base_url}/api/tags", timeout=8.0).json()
                models = tags.get("models") or []
                if models:
                    name = models[0].get("name", "")
            except Exception:
                pass
        n_ctx = 0
        vision = False
        supports_tools = False
        if name:
            try:
                show = self._http.post(
                    f"{self.base_url}/api/show", json={"name": name}, timeout=8.0
                ).json()
                caps = show.get("capabilities", []) or []
                vision = "vision" in caps
                supports_tools = "tools" in caps
                # n_ctx is buried in model_info under a key like "<arch>.context_length"
                for k, v in (show.get("model_info") or {}).items():
                    if k.endswith("context_length"):
                        n_ctx = int(v)
                        break
            except Exception:
                pass
        # n_ctx: prefer the ACTUAL loaded window (/api/ps). model_info.context_length is only the
        # architecture MAX (e.g. 262144), never the real window the model runs with — using it makes the
        # ctx gauge wrong AND defeats auto-compaction (it thinks the budget is huge → never compacts →
        # ollama silently truncates). /api/ps only lists LOADED models, so on a cold start it's empty and
        # we fall back to the arch max; refresh_ollama_nctx() re-reads once the first chat warms the model.
        loaded = self._ollama_loaded_nctx(name or "")
        if loaded:
            n_ctx = loaded
        return ModelInfo(
            backend="ollama",
            name=name,
            n_ctx=n_ctx,
            vision=vision,
            supports_tools=supports_tools,
            base_url=self.base_url,
        )

    @staticmethod
    def _pick_loaded_nctx(ps: dict[str, Any], model: str) -> int:
        """From an /api/ps payload, the context window the named model is loaded with (0 if absent)."""
        for m in ps.get("models") or []:
            if not model or m.get("name") == model or m.get("model") == model:
                cl = m.get("context_length")
                if cl:
                    return int(cl)
        return 0

    def loaded_models(self) -> list[str]:
        """Names of models currently LOADED in ollama (/api/ps); [] on error / non-ollama / none loaded."""
        try:
            ps = self._http.get(f"{self.base_url}/api/ps", timeout=5.0).json()
        except Exception:
            return []
        return [m.get("name", "") for m in (ps.get("models") or []) if m.get("name")]

    def preload_model(self, model: str) -> bool:
        """Load a model into ollama NOW (POST /api/generate with no prompt → it only loads, no generation).
        BLOCKS until the model is in memory. Doing this at startup means n_ctx is read correctly from
        /api/ps and the first real turn isn't slowed by the load. Returns True on success; ollama-only."""
        try:
            r = self._http.post(f"{self.base_url}/api/generate", json={"model": model},
                                 timeout=httpx.Timeout(600.0, connect=15.0))
            return r.status_code == 200
        except Exception:
            return False

    def available_models(self) -> list[str]:
        """Names of ALL models ollama has pulled (/api/tags); [] on error / non-ollama."""
        try:
            tags = self._http.get(f"{self.base_url}/api/tags", timeout=8.0).json()
        except Exception:
            return []
        return [m.get("name", "") for m in (tags.get("models") or []) if m.get("name")]

    def _ollama_loaded_nctx(self, model: str) -> int:
        """The real loaded context window from /api/ps (0 if the model isn't loaded / on error)."""
        try:
            ps = self._http.get(f"{self.base_url}/api/ps", timeout=5.0).json()
        except Exception:
            return 0
        return self._pick_loaded_nctx(ps, model)

    def refresh_ollama_nctx(self) -> bool:
        """Re-read the real loaded window from /api/ps and update self.info.n_ctx. Used after the first
        chat warms the model (cold start: /api/ps was empty at detect time). Returns True if it changed.
        No-op for non-ollama."""
        if self.backend != "ollama" or self.info is None:
            return False
        loaded = self._ollama_loaded_nctx(self.model or "")
        if loaded and loaded != self.info.n_ctx:
            self.info.n_ctx = loaded
            return True
        return False

    # ---- chat -----------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tool_choice: str = "auto",
        enable_thinking: bool | None | object = _UNSET,
        guard_degenerate: bool = False,    # True=switch to streaming, watch for the "repetition spiral" and abort
        degen_grace: float = 40.0,         # don't check for the first N seconds (normal long generation runs)
        on_delta: Any = None,              # callback for each streamed "real content" chunk (for live UI; excludes thinking)
        cancel_check: Any = None,          # called per chunk while streaming; if it returns True → abort (StreamInterrupted)
    ) -> ChatResponse:
        """Send one chat turn; returns a response with reasoning/content/tool_calls separated.

        enable_thinking: omit to use self.enable_thinking; pass True/False to set explicitly;
        pass None to not send the parameter (leave it to the server default).
        guard_degenerate: when on, switch to streaming; after degen_grace seconds the watchdog checks the
        tail's marginal novelty and, if it stays redundant (an unbounded loop), aborts → DegenerateOutput.
        cancel_check: while streaming, called after each chunk; returning True aborts mid-generation
        (closes the connection) and raises StreamInterrupted — this is how ESC interrupts a long generation.
        """
        if self.info is None:
            self.detect()
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        temperature = self.default_temperature if temperature is None else temperature
        think = cast("bool | None", self.enable_thinking if enable_thinking is _UNSET else enable_thinking)

        if self.backend == "anthropic":
            # stream when the caller wants live output (the agent always sets guard_degenerate for that);
            # no repetition watchdog (a frontier model doesn't loop). Non-stream otherwise (e.g. /exit summary).
            if guard_degenerate:
                return self._chat_anthropic_stream(messages, tools, max_tokens, temperature, think,
                                                   on_delta, cancel_check)
            return self._chat_anthropic(messages, tools, max_tokens, temperature, think)
        if guard_degenerate:
            if self.backend == "ollama":
                return self._chat_ollama_stream(messages, tools, max_tokens, temperature,
                                                think, degen_grace, on_delta, cancel_check)
            return self._chat_llamacpp_stream(messages, tools, max_tokens, temperature,
                                              tool_choice, think, degen_grace, on_delta, cancel_check)
        if self.backend == "ollama":
            return self._chat_ollama(messages, tools, max_tokens, temperature, think)
        return self._chat_llamacpp(
            messages, tools, max_tokens, temperature, tool_choice, think
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self._http.post(f"{self.base_url}{path}", json=payload)
        except httpx.HTTPError as e:
            self._log_conn("connection_lost", f"non-stream request {path} failed (no reply / dead connection): {e}")
            raise BackendError(f"Request {path} failed: {e}") from e
        if r.status_code != 200:
            raise BackendError(f"{path} responded {r.status_code}: {r.text[:500]}")
        return r.json()

    def _chat_llamacpp(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        tool_choice: str,
        think: bool | None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repeat_penalty": self.repeat_penalty,   # llama.cpp server accepts its native sampler params here
            "repeat_last_n": self.repeat_last_n,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": think}
        data = self._post("/v1/chat/completions", payload)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        return ChatResponse(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            tool_calls=self._parse_tool_calls(msg.get("tool_calls")),
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage", {}),
            raw=data,
        )

    # ---- Anthropic (Claude) --------------------------------------------
    def _detect_anthropic(self) -> ModelInfo:
        """Build ModelInfo from the provider registry (no probe — cloud has no /health). n_ctx/vision/tools
        come from the fixed table so ctx-metering + auto-compaction work exactly like the local backends."""
        from falamus.core import providers
        p = providers.get("anthropic")
        model = self.model or (p.default_model if p else "claude-opus-4-8")
        self.model = model
        n_ctx = p.n_ctx(model) if p else 200_000
        return ModelInfo(backend="anthropic", name=model, n_ctx=n_ctx,
                         vision=True, supports_tools=True, base_url=self.base_url)

    def list_models(self) -> list[str]:
        """The models this cloud key can use — for the setup picker. [] on any error / non-cloud backend.
        (Provider-generic entry; currently anthropic's GET /v1/models.)"""
        if self.backend != "anthropic":
            return []
        try:
            r = self._http.get(f"{self.base_url}/v1/models", headers=self._anthropic_headers(), timeout=8.0)
            if r.status_code == 200:
                return [m.get("id", "") for m in (r.json().get("data") or []) if m.get("id")]
        except Exception:
            pass
        return []

    @staticmethod
    def _to_anthropic_content(content: Any) -> list[dict[str, Any]]:
        """One OpenAI message's content → a list of Anthropic content blocks (text / image)."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        blocks: list[dict[str, Any]] = []
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                media, _, data = url.partition("base64,")
                media = media[len("data:"):].rstrip(";") if media.startswith("data:") else "image/png"
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": media or "image/png", "data": data or url}})
        return blocks

    @classmethod
    def _to_anthropic_messages(cls, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Rewrite CANONICAL OpenAI messages into Anthropic's shape at the send boundary (like the ollama
        adapter). Returns (system_text, messages). Differences handled:
          - role:system → hoisted OUT into the top-level `system` string (Anthropic has no system message).
          - assistant tool_calls → `tool_use` blocks (arguments JSON-string → object `input`).
          - role:tool → a `role:user` message with a `tool_result` block (tool_call_id → tool_use_id).
          - image_url data-URL → `image` source block (base64 + media_type).
          - consecutive same-role messages are MERGED (Anthropic requires user/assistant to alternate;
            several tool results in a row become one user message with several tool_result blocks)."""
        system_parts: list[str] = []
        built: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                if isinstance(m.get("content"), str) and m["content"].strip():
                    system_parts.append(m["content"])
                continue
            if role == "tool":
                c = m.get("content")
                tr = {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                      "content": c if isinstance(c, str) else cls._to_anthropic_content(c)}
                built.append({"role": "user", "content": [tr]})
                continue
            if role == "assistant":
                blocks = cls._to_anthropic_content(m.get("content"))
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {}) or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                                   "name": fn.get("name", ""), "input": args if isinstance(args, dict) else {}})
                built.append({"role": "assistant", "content": blocks})
                continue
            # user (or anything else) → user with content blocks
            built.append({"role": "user", "content": cls._to_anthropic_content(m.get("content"))})
        # merge consecutive same-role messages
        merged: list[dict[str, Any]] = []
        for msg in built:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"].extend(msg["content"])
            else:
                merged.append(msg)
        return "\n\n".join(system_parts), merged

    @staticmethod
    def _tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """OpenAI tools schema → Anthropic tools ({name, description, input_schema})."""
        out: list[dict[str, Any]] = []
        for t in tools or []:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return out

    def _anthropic_payload(self, messages, tools, max_tokens, think, stream: bool) -> dict[str, Any]:
        """Build the /v1/messages request body (shared by stream + non-stream).

        PROMPT CACHING: the STATIC prefix (system + tools) is marked with cache_control so it's charged once
        (1.25×) then read at 0.1× on every following turn — falamus's system prompt is static by design, a
        big near-free win vs paying full input price for the ~3k-token prefix each turn. temperature is
        deliberately NOT sent (newer Claude models reject it → HTTP 400); Anthropic's default sampling is used.
        """
        system, msgs = self._to_anthropic_messages(messages)
        cap = max_tokens if max_tokens and max_tokens > 0 else self._anthropic_max_out()
        payload: dict[str, Any] = {"model": self.model, "messages": msgs, "max_tokens": cap}
        if stream:
            payload["stream"] = True
        if system:
            payload["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if tools:
            atools = self._tools_to_anthropic(tools)
            atools[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = atools
        if think:
            payload["thinking"] = {"type": "enabled", "budget_tokens": max(1024, cap // 2)}
        return payload

    @staticmethod
    def _anthropic_usage(usage: dict[str, Any], out_tok: int) -> dict[str, Any]:
        """usage dict from Anthropic's counts. prompt_tokens = fresh input + cached (read+creation) so the
        ctx gauge sees the real context size regardless of how much was served from cache."""
        cread = usage.get("cache_read_input_tokens", 0)
        ccreate = usage.get("cache_creation_input_tokens", 0)
        prompt = usage.get("input_tokens", 0) + cread + ccreate
        return {"prompt_tokens": prompt, "completion_tokens": out_tok, "total_tokens": prompt + out_tok,
                "cache_read_input_tokens": cread, "cache_creation_input_tokens": ccreate}

    def _chat_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        think: bool | None,
    ) -> ChatResponse:
        _ = temperature
        payload = self._anthropic_payload(messages, tools, max_tokens, think, stream=False)
        data = self._post_anthropic("/v1/messages", payload)
        content, reasoning = "", ""
        calls: list[ToolCall] = []
        for i, block in enumerate(data.get("content") or []):
            bt = block.get("type")
            if bt == "text":
                content += block.get("text", "")
            elif bt == "thinking":
                reasoning += block.get("thinking", "")
            elif bt == "tool_use":
                calls.append(ToolCall(id=block.get("id") or f"call_{i}",
                                      name=block.get("name", ""),
                                      arguments=block.get("input") or {}))
        usage = data.get("usage", {}) or {}
        return ChatResponse(
            content=content, reasoning=reasoning, tool_calls=calls,
            finish_reason=data.get("stop_reason", ""),
            usage=self._anthropic_usage(usage, usage.get("output_tokens", 0)),
            raw=data,
        )

    def _chat_anthropic_stream(self, messages, tools, max_tokens, temperature, think,
                               on_delta=None, cancel_check=None) -> ChatResponse:
        """Streaming /v1/messages (P2): parse the SSE event stream, feed text deltas to on_delta live, and
        accumulate tool_use input JSON per block. No repetition watchdog (a frontier model doesn't loop);
        cancel_check (ESC) aborts mid-stream → StreamInterrupted."""
        _ = temperature
        payload = self._anthropic_payload(messages, tools, max_tokens, think, stream=True)
        content: list[str] = []
        reasoning: list[str] = []
        tools_acc: dict[int, dict[str, str]] = {}   # index -> {id, name, json}
        usage: dict[str, Any] = {}
        out_tok = 0
        finish = ""
        try:
            with self._http.stream("POST", f"{self.base_url}/v1/messages", json=payload,
                                   headers=self._anthropic_headers()) as r:
                if r.status_code != 200:
                    raise BackendError(f"/v1/messages responded {r.status_code}: {r.read()[:500]!r}")
                for line in r.iter_lines():
                    if cancel_check and cancel_check():          # ESC / server-offline mid-stream → abort
                        raise StreamInterrupted("".join(content))
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    et = ev.get("type")
                    if et == "message_start":
                        usage = (ev.get("message") or {}).get("usage") or {}
                    elif et == "content_block_start":
                        cb = ev.get("content_block") or {}
                        if cb.get("type") == "tool_use":
                            tools_acc[ev.get("index", 0)] = {"id": cb.get("id", ""),
                                                             "name": cb.get("name", ""), "json": ""}
                    elif et == "content_block_delta":
                        d = ev.get("delta") or {}
                        dt = d.get("type")
                        if dt == "text_delta":
                            txt = d.get("text", "")
                            content.append(txt)
                            if on_delta and txt:
                                on_delta(txt)
                        elif dt == "input_json_delta":
                            b = tools_acc.get(ev.get("index", 0))
                            if b is not None:
                                b["json"] += d.get("partial_json", "")
                        elif dt == "thinking_delta":
                            reasoning.append(d.get("thinking", ""))
                    elif et == "message_delta":
                        finish = (ev.get("delta") or {}).get("stop_reason") or finish
                        out_tok = (ev.get("usage") or {}).get("output_tokens", out_tok)
                    elif et == "message_stop":
                        break
        except (StreamInterrupted, BackendError):
            raise
        except Exception as e:  # noqa: BLE001
            self._log_conn("connection_lost", f"anthropic stream failed: {e}")
            raise BackendError(f"Streaming request failed: {e}") from e
        raw_tcs = [{"id": tools_acc[i]["id"] or f"call_{i}",
                    "function": {"name": tools_acc[i]["name"], "arguments": tools_acc[i]["json"]}}
                   for i in sorted(tools_acc)]
        return ChatResponse(
            content="".join(content), reasoning="".join(reasoning),
            tool_calls=self._parse_tool_calls(raw_tcs), finish_reason=finish,
            usage=self._anthropic_usage(usage, out_tok), raw={},
        )

    def _anthropic_max_out(self) -> int:
        from falamus.core import providers
        p = providers.get("anthropic")
        return p.max_out(self.model or "") if p else 8_000

    def _anthropic_headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01",
                "content-type": "application/json"}

    def _post_anthropic(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self._http.post(f"{self.base_url}{path}", json=payload, headers=self._anthropic_headers())
        except httpx.HTTPError as e:
            self._log_conn("connection_lost", f"anthropic {path} failed: {e}")
            raise BackendError(f"Request {path} failed: {e}") from e
        if r.status_code != 200:
            raise BackendError(f"{path} responded {r.status_code}: {r.text[:500]}")
        return r.json()

    @staticmethod
    def _to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rewrite CANONICAL OpenAI-shaped messages into ollama's native /api/chat shape.

        self.messages is always kept OpenAI-shaped (so the context manager / checkpoints see one
        format); ollama differs on two fields and rejects the OpenAI form:
          - assistant tool_calls: function.arguments is a JSON *string* in OpenAI, but ollama wants an
            *object* (else HTTP 400 "Value looks like object, but can't find closing '}' symbol").
          - multimodal content: OpenAI uses a content-block ARRAY ([{type:text}, {type:image_url}]),
            but ollama wants content as a *string* plus a top-level `images` list of RAW base64 (no
            data-URL prefix) — else HTTP 400 "cannot unmarshal array into ... content of type string".
        Originals are not mutated (shallow-copied per message).
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            m2 = dict(m)
            tcs = m2.get("tool_calls")
            if tcs:
                fixed = []
                for tc in tcs:
                    fn = dict(tc.get("function", {}) or {})
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    fn["arguments"] = args
                    fixed.append({"function": fn})
                m2["tool_calls"] = fixed
            content = m2.get("content")
            if isinstance(content, list):
                texts: list[str] = []
                images: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        images.append(url.split("base64,", 1)[-1] if "base64," in url else url)
                m2["content"] = "\n".join(texts)
                if images:
                    m2["images"] = images
            out.append(m2)
        return out

    def _chat_ollama(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        think: bool | None,
    ) -> ChatResponse:
        messages = self._to_ollama_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature,
                        "repeat_penalty": self.repeat_penalty, "repeat_last_n": self.repeat_last_n},
        }
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = think
        data = self._post("/api/chat", payload)
        msg = data.get("message", {})
        return ChatResponse(
            content=msg.get("content") or "",
            reasoning=msg.get("thinking") or "",
            tool_calls=self._parse_tool_calls(msg.get("tool_calls")),
            finish_reason=data.get("done_reason", ""),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            raw=data,
        )

    # ---- streaming generation + degenerate-repetition watchdog ----------
    @staticmethod
    def _watch(produced: list[str], t0: float, state: tuple[float, int], grace: float) -> tuple[float, int]:
        """After `grace` seconds, once a second, measure the tail's marginal novelty; if it stays redundant
        for _DEGEN_SUSTAIN consecutive checks (an UNBOUNDED loop, not a bounded repeat), raise
        DegenerateOutput. state = (last_check_time, consecutive_low_novelty_streak).

        `produced` is the CHUNK LIST; it is joined ONLY when a check actually fires (throttled to ~1/s), not
        once per chunk — so a long generation is O(length) over the run, not O(length²)."""
        last, streak = state
        now = time.time()
        if now - t0 > grace and now - last > 1.0:
            last = now
            text = "".join(produced)
            if _tail_novelty(text) < _DEGEN_RATIO:
                streak += 1
                if streak >= _DEGEN_SUSTAIN:
                    raise DegenerateOutput(
                        f"Degenerate generation: the tail keeps repeating earlier output "
                        f"({len(text)} chars over {int(now - t0)}s); aborted.")
            else:
                streak = 0
        return (last, streak)

    def _chat_llamacpp_stream(
        self, messages, tools, max_tokens, temperature, tool_choice, think,
        grace: float, on_delta=None, cancel_check=None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            "repeat_penalty": self.repeat_penalty, "repeat_last_n": self.repeat_last_n,
            "stream": True, "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": think}

        content: list[str] = []
        reasoning: list[str] = []
        tool_acc: dict[int, dict[str, str]] = {}
        produced: list[str] = []      # combined output of content + tool arguments (for UI streaming + watchdog)
        finish, usage = "", {}
        t0 = time.time()
        wd: tuple[float, int] = (t0, 0)   # degen watchdog state: (last_check_time, low-novelty streak)

        def emit(text: str) -> None:
            produced.append(text)
            if on_delta:
                on_delta(text)

        finished_at: list[float | None] = [None]   # set when finish_reason arrives → starts the [DONE] grace
        stop_wd = threading.Event()
        try:
            with self._http.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as r:
                if r.status_code != 200:
                    raise BackendError(f"/v1/chat/completions responded {r.status_code}: {r.read()[:500]!r}")

                def _await_done() -> None:
                    # the proper end is [DONE]; if it doesn't arrive within _DONE_GRACE after finish_reason
                    # (dead / kept-alive-idle connection), close the socket so the blocked read aborts.
                    while not stop_wd.wait(0.25):
                        ft = finished_at[0]
                        if ft is not None and time.time() - ft > _DONE_GRACE:
                            r.close()
                            return
                threading.Thread(target=_await_done, daemon=True).start()

                for line in r.iter_lines():
                    if cancel_check and cancel_check():    # ESC / server-offline mid-stream → abort now
                        raise StreamInterrupted("".join(produced))
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    ch = (chunk.get("choices") or [{}])[0]
                    delta = ch.get("delta", {}) or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                        emit(delta["content"])
                    rc = delta.get("reasoning_content") or delta.get("reasoning")
                    if rc:
                        reasoning.append(rc)              # thinking channel: not streamed to UI, not watched
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        acc = tool_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            acc["id"] = tc["id"]
                        fn = tc.get("function", {}) or {}
                        if fn.get("name"):
                            acc["name"] = fn["name"]
                        if fn.get("arguments"):
                            acc["args"] += fn["arguments"]
                            emit(fn["arguments"])         # tool args (incl. write_file/deliver content) → live stream + watch
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                        finished_at[0] = time.time()   # keep reading for the trailing usage chunk + [DONE];
                                                       # the watchdog aborts if [DONE] never follows in time.
                    wd = self._watch(produced, t0, wd, grace)
        except (StreamInterrupted, BackendError):
            raise
        except Exception as e:  # noqa: BLE001
            # If the watchdog closed the socket AFTER finish_reason (no [DONE] came), the generation is
            # already complete — that's the expected end, not a failure. A pre-finish error is real.
            waited, got = int(time.time() - t0), len("".join(produced))
            if not finish:
                self._log_conn("connection_lost", f"stream error after {waited}s / {got} chars (no finish_reason — likely a dead/keepalive-dropped connection): {e}")
                raise BackendError(f"Streaming request failed: {e}") from e
            self._log_conn("stream_stall", f"finished but no [DONE] within {_DONE_GRACE:.0f}s → watchdog closed the socket ({got} chars, {waited}s)")
        finally:
            stop_wd.set()
        raw_tcs = [
            {"id": tool_acc[i]["id"], "function": {"name": tool_acc[i]["name"], "arguments": tool_acc[i]["args"]}}
            for i in sorted(tool_acc)
        ]
        return ChatResponse(
            content="".join(content), reasoning="".join(reasoning),
            tool_calls=self._parse_tool_calls(raw_tcs),
            finish_reason=finish, usage=usage, raw={},
        )

    def _chat_ollama_stream(
        self, messages, tools, max_tokens, temperature, think, grace: float,
        on_delta=None, cancel_check=None,
    ) -> ChatResponse:
        messages = self._to_ollama_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature,
                        "repeat_penalty": self.repeat_penalty, "repeat_last_n": self.repeat_last_n},
        }
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = think

        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: list = []
        produced: list[str] = []      # combined output of content + tool arguments (for UI streaming + watchdog)
        finish, usage = "", {}
        t0 = time.time()
        wd: tuple[float, int] = (t0, 0)   # degen watchdog state: (last_check_time, low-novelty streak)

        def emit(text: str) -> None:
            produced.append(text)
            if on_delta:
                on_delta(text)

        try:
            with self._http.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
                if r.status_code != 200:
                    raise BackendError(f"/api/chat responded {r.status_code}: {r.read()[:500]!r}")
                for line in r.iter_lines():
                    if cancel_check and cancel_check():    # ESC / server-offline mid-stream → abort now
                        raise StreamInterrupted("".join(produced))
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message", {}) or {}
                    if msg.get("content"):
                        content.append(msg["content"])
                        emit(msg["content"])
                    if msg.get("thinking"):
                        reasoning.append(msg["thinking"])   # thinking channel: not streamed, not watched
                    if msg.get("tool_calls"):
                        tool_calls = msg["tool_calls"]      # ollama usually sends complete tool_calls at once
                        for tc in msg["tool_calls"]:
                            a = (tc.get("function", {}) or {}).get("arguments", "")
                            emit(a if isinstance(a, str) else json.dumps(a, ensure_ascii=False))
                    if chunk.get("done"):
                        finish = chunk.get("done_reason", "")
                        usage = {
                            "prompt_tokens": chunk.get("prompt_eval_count", 0),
                            "completion_tokens": chunk.get("eval_count", 0),
                        }
                        break   # done (usage is in this same chunk) — stop; don't block on a kept-alive connection
                    wd = self._watch(produced, t0, wd, grace)
        except httpx.HTTPError as e:
            self._log_conn("connection_lost", f"ollama stream failed after {int(time.time()-t0)}s / {len(''.join(produced))} chars: {e}")
            raise BackendError(f"Streaming request failed: {e}") from e
        return ChatResponse(
            content="".join(content), reasoning="".join(reasoning),
            tool_calls=self._parse_tool_calls(tool_calls),
            finish_reason=finish, usage=usage, raw={},
        )

    @staticmethod
    def _parse_tool_calls(raw: Any) -> list[ToolCall]:
        """Normalize tool_calls from both backends; arguments always become a dict."""
        if not raw:
            return []
        calls: list[ToolCall] = []
        for i, tc in enumerate(raw):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}   # fault tolerance: keep the original string
            calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{i}",
                    name=fn.get("name", ""),
                    arguments=args if isinstance(args, dict) else {"_value": args},
                )
            )
        return calls


# ──────────────────────────────────────────────────────────────────────────
# smoke test (connect directly to a known endpoint)
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    print(f"connecting to {base} …")
    with LLMClient(base) as cli:
        info = cli.detect()
        print("detection:", info.summary())

        print("\n[1] basic chat (thinking off by default)")
        r = cli.chat([{"role": "user", "content": "Reply with one sentence: hello."}])
        print("  finish:", r.finish_reason, "| tokens used:", r.usage.get("completion_tokens"))
        print("  reasoning_len:", len(r.reasoning), "| content:", repr(r.content))

        print("\n[1b] same with thinking on (enable_thinking=True)")
        r = cli.chat(
            [{"role": "user", "content": "Reply with one sentence: hello."}],
            enable_thinking=True, max_tokens=2048,
        )
        print("  finish:", r.finish_reason, "| tokens used:", r.usage.get("completion_tokens"))
        print("  reasoning_len:", len(r.reasoning), "| content:", repr(r.content))

        print("\n[2] tool call")
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the current weather in a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "city name"}},
                    "required": ["city"],
                },
            },
        }]
        r = cli.chat(
            [{"role": "user", "content": "What's the weather in Taipei? Use the tool to look it up."}],
            tools=tools,
        )
        print("  finish:", r.finish_reason, "| wants_tool:", r.wants_tool)
        for c in r.tool_calls:
            print(f"  → {c.name}({c.arguments})  id={c.id}")
