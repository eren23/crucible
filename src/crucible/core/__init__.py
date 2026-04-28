"""Core utilities shared across all Crucible modules.

Stable public re-exports — orchestrators and external callers should import
from this module rather than directly from the submodule path.
"""
from crucible.core.doom_loop import detect as detect_doom_loop
from crucible.core.redact import redact_secrets

__all__ = [
    "detect_doom_loop",
    "redact_secrets",
]
