"""LLM client abstraction for the autonomous researcher.

Claude-first with a clean Protocol interface for future expansion.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol



class LLMClient(Protocol):
    """Protocol for LLM backends used by the autonomous researcher."""

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str | None:
        """Send a system + user prompt and return the text response, or None on failure."""
        ...


class AnthropicClient:
    """Anthropic Claude client implementation."""

    def __init__(self, model: str | None = None, max_tokens: int = 4096) -> None:
        self.model = model or os.environ.get("RESEARCHER_MODEL", "claude-sonnet-4-6-20250514")
        self.default_max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str | None:
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.default_max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if response.content and len(response.content) > 0:
                return response.content[0].text
            return None
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            from crucible.core.log import log_warn
            log_warn(f"LLM call failed (model={self.model}): {exc}")
            return None


def parse_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM response text, handling code blocks."""
    text = text.strip()

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Extract from code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Find outermost { ... } braces
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                start = None

    print(f"  WARNING: Could not parse JSON from LLM response ({len(text)} chars)")
    return None
