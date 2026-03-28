"""Secret redaction for trace logs and exports."""
from __future__ import annotations

import re
from typing import Any

# Patterns to redact
_SECRET_KEYS = frozenset({
    "RUNPOD_API_KEY", "WANDB_API_KEY", "ANTHROPIC_API_KEY",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
})

_SECRET_KEY_PATTERNS = re.compile(
    r"(_KEY|_SECRET|_TOKEN|_PASSWORD|_CREDENTIAL)$", re.IGNORECASE
)

_SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),           # Anthropic/OpenAI keys
    re.compile(r"wandb_v1_[a-zA-Z0-9_]{20,}"),     # WandB keys
    re.compile(r"rpd_[a-zA-Z0-9]{20,}"),            # RunPod keys
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),             # HuggingFace tokens
]


def redact_secrets(data: Any) -> Any:
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
    return s
