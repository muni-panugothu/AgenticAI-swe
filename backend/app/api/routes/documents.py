"""Authenticated document lifecycle endpoints for document RAG."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import AuthenticatedUser, get_current_user
from app.services.rag_config import settings
from app.services.rag_document_service import (
    InvalidDocumentCursor,
    RagError,
    delete_document as delete_document_data,
    ingest_document,
    list_documents as list_document_data,
    set_document_shares,
)
from app.services.upload_rate_limit_service import (
    UploadRateLimitExceeded,
    enforce_upload_rate_limit,
)

logger = logging.getLogger("agentic.rag")
router = APIRouter(prefix="/api/documents", tags=["documents"])


class ShareDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_ids: list[str] = Field(default_factory=list, max_length=50)


def _project_scope(user_id: str) -> str | None:
    # Project scope comes from server-side session state, never request JSON.
    from app.services.memory_scope_service import get_active_project_scope

    return get_active_project_scope(user_id).get("project_id")


def _ingest_document_for_user(
    *,
    owner_id: str,
    tenant_id: str | None,
    filename: str | None,
    content_type: str | None,
    raw: bytes,
) -> dict:
    """Resolve project scope and ingest inside one worker thread."""
    project_id = _project_scope(owner_id)
    return ingest_document(
        owner_id=owner_id,
        tenant_id=tenant_id,
        project_id=project_id,
        filename=filename,
        content_type=content_type,
        raw=raw,
    )


def _set_document_shares_for_user(
    *,
    document_id: str,
    owner_id: str,
    tenant_id: str | None,
    shared_with: list[str],
) -> object:
    project_id = _project_scope(owner_id)
    return set_document_shares(
        document_id=document_id,
        owner_id=owner_id,
        tenant_id=tenant_id,
        project_id=project_id,
        shared_with=shared_with,
    )


def _delete_document_for_user(
    *,
    document_id: str,
    owner_id: str,
    tenant_id: str | None,
) -> object:
    project_id = _project_scope(owner_id)
    return delete_document_data(
        document_id=document_id,
        owner_id=owner_id,
        tenant_id=tenant_id,
        project_id=project_id,
    )


@router.get("")
async def list_documents(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict:
    try:
        # Redis project-scope lookup must not block the event loop.
        project_id = await run_in_threadpool(
            _project_scope,
            user.id,
        )

        result = await run_in_threadpool(
            list_document_data,
            user_id=user.id,
            tenant_id=user.tenant_id,
            project_id=project_id,
            page_size=limit,
            cursor=cursor,
        )

    except InvalidDocumentCursor as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_DOCUMENT_CURSOR",
                "error": str(exc),
            },
        ) from exc

    except Exception as exc:
        logger.warning(
            "RAG document listing failed (%s)",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "RAG_STORE_UNAVAILABLE",
                "error": "Document store unavailable",
            },
        ) from exc

    return result


@router.post("/upload")
@router.post("/ingest")
async def upload_document(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    file: Annotated[UploadFile, File(...)],
) -> dict:
    try:
        try:
            await run_in_threadpool(
                enforce_upload_rate_limit,
                user.id,
            )
        except UploadRateLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "UPLOAD_RATE_LIMIT_EXCEEDED",
                    "error": "Upload rate limit exceeded",
                    "retry_after_seconds": exc.retry_after_seconds,
                },
                headers={"Retry-After": str(exc.retry_after_seconds)},
            ) from exc
        except Exception as exc:
            logger.warning(
                "Upload rate-limit check failed (%s)",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "UPLOAD_RATE_LIMIT_UNAVAILABLE",
                    "error": "Upload service temporarily unavailable",
                },
            ) from exc

        # A single bounded read prevents unbounded memory use for multipart uploads.
        raw = await file.read(settings.max_upload_bytes + 1)

        try:
            return await run_in_threadpool(
                _ingest_document_for_user,
                owner_id=user.id,
                tenant_id=user.tenant_id,
                filename=file.filename,
                content_type=file.content_type,
                raw=raw,
            )
        except RagError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "DOCUMENT_REJECTED",
                    "error": str(exc),
                },
            ) from exc
        except Exception as exc:
            logger.warning(
                "RAG ingestion failed (%s)",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "RAG_INGESTION_FAILED",
                    "error": "Document ingestion failed",
                },
            ) from exc
    finally:
        await file.close()


@router.put("/{document_id}/share")
async def share_document(
    document_id: Annotated[str, Path(pattern=r"^[a-f0-9]{64}$")],
    body: ShareDocumentRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    if user.tenant_id is None and body.user_ids:
        raise HTTPException(status_code=403, detail={"code": "SHARING_REQUIRES_TENANT", "error": "Sharing requires an explicit tenant"})
    from app.services.auth_service import get_user_by_id

    for user_id in set(body.user_ids):
        target = get_user_by_id(user_id)
        if not target or target.get("status") != "active" or target.get("tenant_id") != user.tenant_id:
            raise HTTPException(status_code=400, detail={"code": "INVALID_SHARE_TARGET", "error": "Every recipient must be an active user in the same tenant"})
    try:
        updated = await run_in_threadpool(
            _set_document_shares_for_user,
            document_id=document_id,
            owner_id=user.id,
            tenant_id=user.tenant_id,
            shared_with=body.user_ids,
        )
    except Exception as exc:
        logger.warning("RAG share update failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail={"code": "RAG_STORE_UNAVAILABLE", "error": "Document store unavailable"}) from exc
    if not updated:
        raise HTTPException(status_code=404, detail={"code": "DOCUMENT_NOT_FOUND", "error": "Document not found"})
    return {"document_id": document_id, "shared_with": sorted(set(body.user_ids)), "permission_level": "shared_read" if body.user_ids else "private"}


@router.delete("/{document_id}")
async def delete_document(
    document_id: Annotated[str, Path(pattern=r"^[a-f0-9]{64}$")],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    try:
        deleted = await run_in_threadpool(
            _delete_document_for_user,
            document_id=document_id,
            owner_id=user.id,
            tenant_id=user.tenant_id,
        )
    except Exception as exc:
        logger.warning("RAG document deletion failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail={"code": "RAG_STORE_UNAVAILABLE", "error": "Document store unavailable"}) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": "DOCUMENT_NOT_FOUND", "error": "Document not found"})
    return {"document_id": document_id, "deleted_chunks": deleted}
