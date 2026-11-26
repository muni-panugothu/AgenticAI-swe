
"""Streaming request-size enforcement for document uploads."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable


class UploadBodyTooLarge(Exception):
    """Raised when an upload request exceeds its configured body limit."""


class UploadBodyLimitMiddleware:
    """Enforce upload body limits without buffering the entire request."""

    UPLOAD_PATHS = {
        "/api/documents/upload",
        "/api/documents/ingest",
    }

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        max_body_bytes: int,
    ) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")

        self.app = app
        self.max_body_bytes = max_body_bytes

    async def _reject(
        self,
        send: Callable[..., Awaitable[None]],
        *,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        body = json.dumps(
            {
                "detail": {
                    "code": code,
                    "error": message,
                }
            }
        ).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })

        await send({
            "type": "http.response.body",
            "body": body,
        })

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[dict[str, Any]]],
        send: Callable[..., Awaitable[None]],
    ) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self.UPLOAD_PATHS
        ):
            await self.app(scope, receive, send)
            return

        content_lengths = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"content-length"
        ]

        if len(content_lengths) > 1:
            await self._reject(
                send,
                status_code=400,
                code="INVALID_CONTENT_LENGTH",
                message="Multiple Content-Length headers are not allowed",
            )
            return

        if content_lengths:
            raw_length = content_lengths[0]

            if not raw_length or not raw_length.isdigit():
                await self._reject(
                    send,
                    status_code=400,
                    code="INVALID_CONTENT_LENGTH",
                    message="Invalid Content-Length header",
                )
                return

            declared_length = int(raw_length)

            if declared_length > self.max_body_bytes:
                await self._reject(
                    send,
                    status_code=413,
                    code="UPLOAD_REQUEST_TOO_LARGE",
                    message="Upload request exceeds the allowed size",
                )
                return

        received_bytes = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received_bytes

            message = await receive()

            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))

                if received_bytes > self.max_body_bytes:
                    raise UploadBodyTooLarge

            return message

        try:
            await self.app(scope, limited_receive, send)
        except UploadBodyTooLarge:
            await self._reject(
                send,
                status_code=413,
                code="UPLOAD_REQUEST_TOO_LARGE",
                message="Upload request exceeds the allowed size",
            )
