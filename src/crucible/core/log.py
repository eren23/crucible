"""Structured logging helpers for Crucible CLI output."""
from __future__ import annotations

import sys
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _print(prefix: str, msg: str, *, file: object = None) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {prefix} {msg}", file=file or sys.stderr, flush=True)


def log_info(msg: str) -> None:
    _print("INFO", msg)


def log_step(msg: str) -> None:
    _print(">>>", msg)


def log_success(msg: str) -> None:
    _print("OK", msg)


def log_warn(msg: str) -> None:
    _print("WARN", msg)


def log_error(msg: str) -> None:
    _print("ERROR", msg, file=sys.stderr)
