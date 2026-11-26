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

from app.api.routes import auth, approvals, chat, documents, health
from app.api.routes.auth import limiter
from app.core.config import get_settings
from app.core.middleware import UploadBodyLimitMiddleware
from app.services.auth_service import ensure_auth_indexes
from app.services.rag_config import settings as rag_settings


logger = logging.getLogger("agentic.api")


def create_app() -> FastAPI:
    settings = get_settings()
    settings.validate_startup()

    app = FastAPI(
        title="Agentic AI Software Engineer API",
        version="1.0.0",
    )
    app.add_middleware(
        UploadBodyLimitMiddleware,
        max_body_bytes=rag_settings.max_upload_bytes + (64 * 1024),
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    origins = settings.cors_origins or [
        "http://localhost:8787",
        "http://127.0.0.1:8787",
        "http://localhost:8081",
        "http://localhost:19006",
        # Vite web app (web/)
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # LAN / Vite network URL (e.g. http://192.168.x.x:5173)
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token", "Accept"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ):
        # Pydantic error "ctx" may contain non-JSON-serializable objects.
        safe_details = []
        for err in exc.errors():
            item = {k: v for k, v in err.items() if k != "ctx"}
            ctx = err.get("ctx")
            if isinstance(ctx, dict):
                item["ctx"] = {
                    key: (str(val) if not isinstance(val, (str, int, float, bool, type(None))) else val)
                    for key, val in ctx.items()
                }
            safe_details.append(item)
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation failed",
                "code": "VALIDATION_ERROR",
                "details": safe_details,
            },
        )

    @app.on_event("startup")
    def _startup():
        try:
            ensure_auth_indexes()
        except Exception as exc:
            logger.warning("Auth index setup skipped: %s", exc)
        try:
            from app.services.approval_service import ensure_approval_indexes
            from app.services.tool_permission_service import ensure_permission_indexes

            ensure_approval_indexes()
            ensure_permission_indexes()
        except Exception as exc:
            logger.warning("Agent security index setup skipped: %s", exc)

    @app.on_event("shutdown")
    def _shutdown():
        from app.services.resource_lifecycle import shutdown_application_resources

        try:
            shutdown_application_resources()
        except Exception as exc:
            logger.warning("Resource shutdown failed: %s", exc)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(approvals.router)
    return app


app = create_app()
