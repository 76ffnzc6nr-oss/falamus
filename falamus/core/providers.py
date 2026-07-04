"""Cloud-provider registry (internal, fixed).

The single source of truth for which cloud AI providers falamus supports and how to reach them. The
`canonical id` (dict key) is managed HERE, never typed by the user — the api-key setup and backend
picker present these as a menu, so multi-provider naming stays consistent and always maps to the right
endpoint / model table.

Adding a provider later (OpenAI, Gemini, …) = adding one entry here; the storage layer (secrets.py),
the entry flow, and the context/agent layers don't change.

`models` holds per-model context window (n_ctx) and output cap (max_out) — the API's model-list endpoint
usually returns only names, so these come from here (unknown models fall back to the provider default).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    id: str
    display: str
    endpoint: str
    default_model: str
    models: dict[str, tuple[int, int]]   # model name -> (n_ctx, max_out)
    default_ctx: int = 200_000
    default_max_out: int = 8_000
    auth: str = "anthropic"              # header style (see client.py)

    def n_ctx(self, model: str) -> int:
        return self.models.get(model, (self.default_ctx, self.default_max_out))[0]

    def max_out(self, model: str) -> int:
        return self.models.get(model, (self.default_ctx, self.default_max_out))[1]


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        id="anthropic",
        display="Anthropic (Claude)",
        endpoint="https://api.anthropic.com",
        default_model="claude-opus-4-8",
        models={
            "claude-opus-4-8": (200_000, 32_000),
            "claude-sonnet-4-6": (200_000, 64_000),
            "claude-haiku-4-5-20251001": (200_000, 32_000),
        },
    ),
}


def is_cloud(backend: str) -> bool:
    return backend in PROVIDERS


def get(provider: str) -> Provider | None:
    return PROVIDERS.get(provider)


def ids() -> list[str]:
    return list(PROVIDERS)
