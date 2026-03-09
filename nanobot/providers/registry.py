"""
Provider Registry — single source of truth for LLM provider metadata.

Adding a new provider:
  1. Add a ProviderSpec to PROVIDERS below.
  2. Add a field to ProvidersConfig in config/schema.py.
  Done. Env vars, prefixing, config matching, status display all derive from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderSpec:
    """One LLM provider's metadata.

    Placeholders in env_extras values:
      {api_key}  — the user's API key
      {api_base} — api_base from config, or this spec's default_api_base
    """

    # identity
    name: str  # config field name, e.g. "gemini"
    keywords: tuple[str, ...]  # model-name keywords for matching (lowercase)
    env_key: str  # LiteLLM env var, e.g. "GEMINI_API_KEY"
    display_name: str = ""  # shown in `nanobot status`

    # model prefixing
    litellm_prefix: str = ""  # "gemini" → model becomes "gemini/{model}"
    skip_prefixes: tuple[str, ...] = ()  # don't prefix if model already starts with these

    # extra env vars
    env_extras: tuple[tuple[str, str], ...] = ()

    # gateway / local detection
    is_gateway: bool = False
    is_local: bool = False
    detect_by_key_prefix: str = ""
    detect_by_base_keyword: str = ""
    default_api_base: str = ""

    # gateway behavior
    strip_model_prefix: bool = False

    # per-model param overrides
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()

    # OAuth-based providers don't use API keys
    is_oauth: bool = False

    # Direct providers bypass LiteLLM entirely
    is_direct: bool = False

    # Provider supports cache_control on content blocks
    supports_prompt_caching: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.name.title()


# ---------------------------------------------------------------------------
# PROVIDERS — the registry.
# ---------------------------------------------------------------------------

PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="gemini",
        keywords=("gemini",),
        env_key="GEMINI_API_KEY",
        display_name="Gemini",
        litellm_prefix="gemini",
        skip_prefixes=("gemini/",),
    ),
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def find_by_model(model: str) -> ProviderSpec | None:
    """Match a standard provider by model-name keyword (case-insensitive)."""
    model_lower = model.lower()
    model_normalized = model_lower.replace("-", "_")
    model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
    normalized_prefix = model_prefix.replace("-", "_")
    std_specs = [s for s in PROVIDERS if not s.is_gateway and not s.is_local]

    for spec in std_specs:
        if model_prefix and normalized_prefix == spec.name:
            return spec

    for spec in std_specs:
        if any(
            kw in model_lower or kw.replace("-", "_") in model_normalized for kw in spec.keywords
        ):
            return spec
    return None


def find_gateway(
    provider_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ProviderSpec | None:
    """Detect gateway/local provider."""
    if provider_name:
        spec = find_by_name(provider_name)
        if spec and (spec.is_gateway or spec.is_local):
            return spec

    for spec in PROVIDERS:
        if spec.detect_by_key_prefix and api_key and api_key.startswith(spec.detect_by_key_prefix):
            return spec
        if spec.detect_by_base_keyword and api_base and spec.detect_by_base_keyword in api_base:
            return spec

    return None


def find_by_name(name: str) -> ProviderSpec | None:
    """Find a provider spec by config field name."""
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None
