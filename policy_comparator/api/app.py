"""FastAPI application factory.

Serves the API and the static frontend from the same origin, which means the
browser never makes a cross-origin call and there is no CORS surface to widen.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..crypto import EncryptionNotConfigured
from . import routes_auth, routes_providers, routes_quotes

logger = logging.getLogger("policy_comparator")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Policy Comparator",
        description=(
            "Internal staff tool for requesting and comparing Italian auto-insurance "
            "quotations across multiple providers."
        ),
        version="0.1.0",
    )

    app.include_router(routes_auth.router)
    app.include_router(routes_quotes.router)
    app.include_router(routes_providers.router)

    @app.get("/api/health", tags=["system"])
    def health() -> dict:
        return {
            "status": "ok",
            "mode": settings.mode,
            "live_provider_automation": settings.live_provider_automation,
            "demonstration_mode": all(
                settings.provider(pid).is_mock for pid in settings.providers
            ),
        }

    @app.exception_handler(EncryptionNotConfigured)
    def _encryption_error(_: Request, exc: EncryptionNotConfigured) -> JSONResponse:
        # Surfaced explicitly rather than as a 500: it is always a deployment
        # configuration problem with a specific fix.
        logger.error("encryption is not configured: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Encryption is not configured on this deployment."},
        )

    if FRONTEND_DIR.is_dir():
        app.mount(
            "/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets"
        )

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
