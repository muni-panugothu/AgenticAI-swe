"""Document API placeholders — authenticated; RAG Topics 22+ not implemented."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthenticatedUser, get_current_user

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _rag_not_ready() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "error": "Document RAG is not enabled yet (Topics 22–32 pending).",
            "code": "RAG_UNAVAILABLE",
        },
    )


@router.get("")
async def list_documents(
    _user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict[str, list[Any]]:
    return {"documents": []}


@router.post("/upload")
async def upload_document(
    _user: Annotated[AuthenticatedUser, Depends(get_current_user)],
):
    raise _rag_not_ready()


@router.post("/ingest")
async def ingest_document(
    _user: Annotated[AuthenticatedUser, Depends(get_current_user)],
):
    raise _rag_not_ready()
