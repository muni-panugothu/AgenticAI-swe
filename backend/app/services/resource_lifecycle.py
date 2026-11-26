"""Shared resource shutdown — MongoDB, Groq, Gemini HTTP clients."""

from __future__ import annotations

import logging

logger = logging.getLogger("agentic.resources")


def shutdown_application_resources() -> None:
    """Close long-lived clients created at import time."""
    try:
        from app.services.memory_service import close_mongo_client

        close_mongo_client()
        logger.info("MongoDB client closed")
    except Exception as exc:
        # Driver exceptions may include deployment addresses or URI details.
        logger.warning("MongoDB shutdown failed (%s)", type(exc).__name__)

    try:
        from app import chatbot

        groq_client = getattr(chatbot, "client", None)
        if groq_client is not None:
            close_fn = getattr(groq_client, "close", None)
            if callable(close_fn):
                close_fn()
            elif hasattr(groq_client, "_client") and hasattr(
                groq_client._client, "close"
            ):
                groq_client._client.close()
        logger.info("Groq client closed")
    except Exception as exc:
        logger.warning("Groq shutdown failed (%s)", type(exc).__name__)

    try:
        from app.services.embedding_service import client as embedding_client

        close_fn = getattr(embedding_client, "close", None)
        if callable(close_fn):
            close_fn()
            logger.info("Gemini embedding client closed")
    except Exception as exc:
        logger.warning("Gemini embedding client shutdown failed (%s)", type(exc).__name__)

    try:
        from app.services.embedding_service import close_rag_client

        close_rag_client()
        logger.info("Gemini RAG embedding client closed")
    except Exception as exc:
        logger.warning("Gemini RAG embedding client shutdown failed (%s)", type(exc).__name__)

    try:
        from app.services.temporary_memory_service import redis as redis_client

        close_fn = getattr(redis_client, "close", None)
        if callable(close_fn):
            close_fn()
            logger.info("Redis client closed")
    except Exception as exc:
        logger.warning("Redis shutdown failed (%s)", type(exc).__name__)
