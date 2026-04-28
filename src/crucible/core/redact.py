"""Secret redaction for trace logs and exports."""
from __future__ import annotations

import re

from crucible.core.types import JsonValue

# Patterns to redact
_SECRET_KEYS = frozenset({
    "RUNPOD_API_KEY", "WANDB_API_KEY", "ANTHROPIC_API_KEY",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
})

_SECRET_KEY_PATTERNS = re.compile(
    r"(_KEY|_SECRET|_TOKEN|_PASSWORD|_CREDENTIAL)$", re.IGNORECASE
)

# Order matters: more-specific patterns (e.g. sk-ant-) first so they don't get
# partially-matched by a generic sk- pattern.
_SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),                # Anthropic (with dashes)
    re.compile(r"sk-[a-zA-Z0-9_\-]{15,}"),                    # OpenAI / generic
    re.compile(r"wandb_v\d+_[a-zA-Z0-9_]{15,}"),              # WandB keys (any version)
    re.compile(r"rpd_[a-zA-Z0-9]{15,}"),                      # RunPod keys
    re.compile(r"hf_[a-zA-Z0-9]{15,}"),                       # HuggingFace tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),              # GitHub fine-grained PAT
    re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}"),  # GitHub classic PATs
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),             # AWS access keys
    re.compile(r"\b[Bb]earer\s+[A-Za-z0-9._\-=]{20,}"),       # Bearer tokens
]

# Env-style assignment: KEY=value where key looks secret-y. Captures the key name
# (group 1) so we keep it visible while redacting the value (group 2). Handles
# `WANDB_API_KEY=foo`, `export HF_TOKEN=bar`, `env: GITHUB_TOKEN=baz`.
_ENV_ASSIGNMENT_RE = re.compile(
    r"(\b\w*(?:TOKEN|KEY|PASSWORD|SECRET|CREDENTIAL)\w*)"
    r"\s*[:=]\s*"
    r"([^\s'\";]+)",
    re.IGNORECASE,
)


def redact_secrets(data: JsonValue) -> JsonValue:
    """Deep-copy data with secret values replaced by '<REDACTED>'."""
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if isinstance(k, str) and (_is_secret_key(k)):
                result[k] = "<REDACTED>"
            else:
                result[k] = redact_secrets(v)
        return result
    elif isinstance(data, list):
        return [redact_secrets(item) for item in data]
    elif isinstance(data, str):
        return _redact_string(data)
    return data


def _is_secret_key(key: str) -> bool:
    return key in _SECRET_KEYS or bool(_SECRET_KEY_PATTERNS.search(key))


def _redact_string(s: str) -> str:
    for pattern in _SECRET_VALUE_PATTERNS:
        s = pattern.sub("<REDACTED>", s)
    s = _ENV_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=<REDACTED>", s)
    return s
