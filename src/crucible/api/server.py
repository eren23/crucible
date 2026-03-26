"""Crucible API server — lightweight REST interface for experiment tracking.

Start via CLI:
    crucible serve --port 8741

Or directly:
    python -m crucible.api.server
"""
from __future__ import annotations

from crucible import __version__


def create_app():
    """Create FastAPI app. Lazy-imports fastapi."""
    from fastapi import FastAPI

    from crucible.api.auth import api_key_middleware
    from crucible.api.routes import router

    app = FastAPI(
        title="Crucible API",
        version=__version__,
        description="Lightweight API for ML experiment tracking",
    )
    app.middleware("http")(api_key_middleware)
    app.include_router(router, prefix="/api/v1")
    return app


def main(port: int = 8741, host: str = "0.0.0.0") -> None:
    """Run the API server with uvicorn."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
