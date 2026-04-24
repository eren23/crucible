"""Model adapters for the Diff-XYZ harness.

Protocol:
    generate(system_prompt: str, user_prompt: str, *, max_tokens: int = 4096,
             temperature: float = 0.0) -> str

Backends:
  - anthropic:<model_id>    via `anthropic` SDK
  - openai:<model_id>       via `openai` SDK
  - google:<model_id>       via `google-generativeai` SDK
  - hf:<repo_id>            via local transformers + vllm/torch (stub)
  - dummy:<mode>            deterministic echo for tests

Model strings are parsed by `resolve_backend`. All SDKs are lazy-imported so
the test suite doesn't need optional deps.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class ModelError(Exception):
    """Raised when a model adapter fails for a non-recoverable reason."""


class ModelBackend(Protocol):
    """Minimal generation protocol every adapter satisfies."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Dummy backend — used by tests and for dry-runs without API keys
# ---------------------------------------------------------------------------


@dataclass
class DummyBackend:
    """Returns deterministic canned output keyed by `mode`.

    Modes:
      - "echo":         returns the user prompt verbatim (useful for parse tests)
      - "empty":        returns ""
      - "reference":    returns the substring between '--- Diff' markers
                        (hack for reproducing "copy the diff" behaviour)
    """
    mode: str = "echo"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        del system_prompt, max_tokens, temperature
        if self.mode == "echo":
            return user_prompt
        if self.mode == "empty":
            return ""
        if self.mode == "reference":
            # For Diff-XYZ: if the prompt contains "--- Diff (fmt) ---" blocks,
            # extract and return them verbatim. Used in synthetic eval tests.
            start = user_prompt.find("--- Diff (")
            if start < 0:
                return ""
            end = user_prompt.find("--- End diff ---", start)
            if end < 0:
                return ""
            chunk = user_prompt[start:end]
            first_newline = chunk.find("\n")
            return chunk[first_newline + 1 :] if first_newline >= 0 else chunk
        raise ModelError(f"unknown dummy mode {self.mode!r}")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@dataclass
class AnthropicBackend:
    """Adapter for anthropic.Messages (Claude 3/4.x)."""

    model_id: str
    api_key: str | None = None
    max_retries: int = 3

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ModelError("anthropic SDK not installed. `pip install anthropic`.") from exc

        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ModelError("ANTHROPIC_API_KEY is not set.")

        client = anthropic.Anthropic(api_key=key)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = client.messages.create(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return "".join(
                    block.text for block in msg.content if getattr(block, "type", "") == "text"
                )
            except Exception as exc:  # noqa: BLE001 — we classify below
                last_err = exc
                if _should_retry(exc, attempt, self.max_retries):
                    time.sleep(2 ** attempt)
                    continue
                raise ModelError(f"anthropic.generate failed: {exc}") from exc
        raise ModelError(f"anthropic.generate exhausted {self.max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@dataclass
class OpenAIBackend:
    """Adapter for openai.Chat.completions (GPT-4o, GPT-4.1, GPT-5, etc.)."""

    model_id: str
    api_key: str | None = None
    max_retries: int = 3

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ModelError("openai SDK not installed. `pip install openai`.") from exc

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ModelError("OPENAI_API_KEY is not set.")

        client = openai.OpenAI(api_key=key)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc
                if _should_retry(exc, attempt, self.max_retries):
                    time.sleep(2 ** attempt)
                    continue
                raise ModelError(f"openai.generate failed: {exc}") from exc
        raise ModelError(f"openai.generate exhausted {self.max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Google (Gemini)
# ---------------------------------------------------------------------------


@dataclass
class GoogleBackend:
    """Adapter for google-generativeai (Gemini 1.5 / 2.x / 2.5)."""

    model_id: str
    api_key: str | None = None
    max_retries: int = 3

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ModelError(
                "google-generativeai SDK not installed. `pip install google-generativeai`."
            ) from exc

        key = self.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ModelError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set.")
        genai.configure(api_key=key)

        model = genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=system_prompt if system_prompt else None,
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = model.generate_content(
                    user_prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                return resp.text or ""
            except Exception as exc:
                last_err = exc
                if _should_retry(exc, attempt, self.max_retries):
                    time.sleep(2 ** attempt)
                    continue
                raise ModelError(f"google.generate failed: {exc}") from exc
        raise ModelError(f"google.generate exhausted {self.max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Local HF backend — stub (proper vLLM/transformers wiring deferred)
# ---------------------------------------------------------------------------


@dataclass
class HFBackend:
    """Stub for local HuggingFace models (Qwen, DeepSeek-Coder, etc.).

    The full implementation belongs in a follow-up commit — it requires
    transformers + accelerate + quantization, none of which belong in core's
    default dep tree. For now, raises ModelError with an installation hint.
    """

    model_id: str

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        del system_prompt, user_prompt, max_tokens, temperature
        raise ModelError(
            f"hf:{self.model_id} backend not yet wired; run via `crucible notebook "
            f"export --runtime colab-h100` or use an API backend for now."
        )


# ---------------------------------------------------------------------------
# Resolver + factory
# ---------------------------------------------------------------------------


_BACKEND_FACTORIES: dict[str, Callable[[str], ModelBackend]] = {
    "anthropic": lambda mid: AnthropicBackend(model_id=mid),
    "openai": lambda mid: OpenAIBackend(model_id=mid),
    "google": lambda mid: GoogleBackend(model_id=mid),
    "hf": lambda mid: HFBackend(model_id=mid),
    "dummy": lambda mid: DummyBackend(mode=mid),
}


def resolve_backend(model_spec: str) -> ModelBackend:
    """Parse `backend:model_id` into a ModelBackend instance.

    Examples:
        anthropic:claude-sonnet-4-6
        openai:gpt-4.1-mini
        google:gemini-2.5-flash
        dummy:echo
    """
    if ":" not in model_spec:
        raise ModelError(
            f"model_spec must be 'backend:model_id' (got {model_spec!r}). "
            f"Known backends: {sorted(_BACKEND_FACTORIES)}."
        )
    backend, _, model_id = model_spec.partition(":")
    backend = backend.strip().lower()
    model_id = model_id.strip()
    if backend not in _BACKEND_FACTORIES:
        raise ModelError(
            f"unknown backend {backend!r}. Known: {sorted(_BACKEND_FACTORIES)}."
        )
    if not model_id:
        raise ModelError(f"empty model_id in spec {model_spec!r}.")
    return _BACKEND_FACTORIES[backend](model_id)


# ---------------------------------------------------------------------------
# Retry classifier
# ---------------------------------------------------------------------------


_RETRYABLE_SUBSTRINGS = (
    "rate limit",
    "overloaded",
    "timeout",
    "timed out",
    "502",
    "503",
    "504",
    "connection",
)


def _should_retry(exc: Exception, attempt: int, max_retries: int) -> bool:
    """Retry on rate-limit / transient network errors, not on 4xx logic errors."""
    if attempt >= max_retries - 1:
        return False
    msg = str(exc).lower()
    return any(needle in msg for needle in _RETRYABLE_SUBSTRINGS)


def record_usage(backend: ModelBackend) -> dict[str, Any]:
    """Stub for future token-accounting; returns empty dict today."""
    del backend
    return {}
