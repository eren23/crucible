"""Tests for evaluation/diff_xyz/models.py (backend resolution + dummy only).

API-backed adapters are covered by the end-to-end smoke test that runs against
live keys, not here.
"""
from __future__ import annotations

import pytest

from evaluation.diff_xyz.models import (
    AnthropicBackend,
    DummyBackend,
    GoogleBackend,
    HFBackend,
    ModelError,
    OpenAIBackend,
    resolve_backend,
)


# ---------------------------------------------------------------------------
# resolve_backend parsing
# ---------------------------------------------------------------------------


def test_resolve_anthropic():
    b = resolve_backend("anthropic:claude-sonnet-4-6")
    assert isinstance(b, AnthropicBackend)
    assert b.model_id == "claude-sonnet-4-6"


def test_resolve_openai():
    b = resolve_backend("openai:gpt-4.1-mini")
    assert isinstance(b, OpenAIBackend)
    assert b.model_id == "gpt-4.1-mini"


def test_resolve_google():
    b = resolve_backend("google:gemini-2.5-flash")
    assert isinstance(b, GoogleBackend)
    assert b.model_id == "gemini-2.5-flash"


def test_resolve_hf():
    b = resolve_backend("hf:Qwen/Qwen2.5-Coder-7B-Instruct")
    assert isinstance(b, HFBackend)
    assert b.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_resolve_dummy():
    b = resolve_backend("dummy:echo")
    assert isinstance(b, DummyBackend)
    assert b.mode == "echo"


def test_resolve_no_colon_raises():
    with pytest.raises(ModelError, match="backend:model_id"):
        resolve_backend("claude-sonnet-4-6")


def test_resolve_unknown_backend_raises():
    with pytest.raises(ModelError, match="unknown backend"):
        resolve_backend("bedrock:claude-sonnet-4-6")


def test_resolve_empty_model_id_raises():
    with pytest.raises(ModelError, match="empty model_id"):
        resolve_backend("anthropic:")


# ---------------------------------------------------------------------------
# DummyBackend behaviours
# ---------------------------------------------------------------------------


def test_dummy_echo_returns_user_prompt():
    b = DummyBackend(mode="echo")
    assert b.generate("sys", "hello world") == "hello world"


def test_dummy_empty_returns_empty():
    b = DummyBackend(mode="empty")
    assert b.generate("sys", "anything") == ""


def test_dummy_reference_extracts_diff_block():
    b = DummyBackend(mode="reference")
    prompt = (
        "header\n"
        "--- Diff (udiff) ---\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
        "--- End diff ---\n"
        "tail"
    )
    out = b.generate("sys", prompt)
    assert "@@ -1,1 +1,1 @@" in out
    assert "-old" in out
    assert "+new" in out


def test_dummy_reference_no_diff_returns_empty():
    b = DummyBackend(mode="reference")
    assert b.generate("sys", "no diff markers here") == ""


def test_dummy_unknown_mode_raises():
    b = DummyBackend(mode="bogus")
    with pytest.raises(ModelError, match="unknown dummy mode"):
        b.generate("sys", "usr")


# ---------------------------------------------------------------------------
# Missing API key guards
# ---------------------------------------------------------------------------


def test_anthropic_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    b = AnthropicBackend(model_id="claude-sonnet-4-6")
    # Only triggers if anthropic is actually installed — otherwise the import
    # error fires first, which is a different ModelError but still blocks.
    with pytest.raises(ModelError):
        b.generate("sys", "usr")


def test_openai_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    b = OpenAIBackend(model_id="gpt-4.1-mini")
    with pytest.raises(ModelError):
        b.generate("sys", "usr")


def test_hf_backend_raises_not_implemented():
    b = HFBackend(model_id="Qwen/Qwen2.5-Coder-7B-Instruct")
    with pytest.raises(ModelError, match="not yet wired"):
        b.generate("sys", "usr")
