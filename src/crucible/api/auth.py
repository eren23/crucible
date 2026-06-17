"""API key authentication middleware for Crucible API."""
from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request, Response


async def api_key_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Check CRUCIBLE_API_KEY if set. When unset, all requests pass through."""
    expected_key = os.environ.get("CRUCIBLE_API_KEY", "")
    if expected_key:
        from fastapi.responses import JSONResponse

        provided = request.headers.get("X-API-Key", "")
        if provided != expected_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
    return await call_next(request)
