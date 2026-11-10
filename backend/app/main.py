"""FastAPI entry point — auth, chat, health."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import auth, chat, documents, health
from app.api.routes.auth import limiter
from app.core.config import get_settings
from app.services.auth_service import ensure_auth_indexes

logger = logging.getLogger("agentic.api")


def create_app() -> FastAPI:
    settings = get_settings()
    settings.validate_startup()

    app = FastAPI(
        title="Agentic AI Software Engineer API",
        version="1.0.0",
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    origins = settings.cors_origins or [
        "http://localhost:8787",
        "http://127.0.0.1:8787",
        "http://localhost:8081",
        "http://localhost:19006",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation failed",
                "code": "VALIDATION_ERROR",
                "details": exc.errors(),
            },
        )

    @app.on_event("startup")
    def _startup():
        try:
            ensure_auth_indexes()
        except Exception as exc:
            logger.warning("Auth index setup skipped: %s", exc)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    return app


app = create_app()
