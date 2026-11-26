import atexit
import os
import math
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GOOGLE API CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY is not configured in .env"
    )


# ============================================================
# GOOGLE CLIENT
# ============================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)
atexit.register(client.close)
rag_client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options=types.HttpOptions(timeout=15000),
)
atexit.register(rag_client.close)


def close_rag_client():
    """Close the RAG-specific bounded-timeout Gemini client on app shutdown."""
    try:
        rag_client.close()
    except Exception:
        pass


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"


def create_rag_embeddings(
    texts,
    *,
    task_type,
    dimensions=768,
    batch_size=32,
    model=EMBEDDING_MODEL,
    embedding_version=None,
    chunking_version=None,
    use_cache=None,
):
    """Create bounded, validated RAG embeddings without changing memory vectors.

    Atlas currently configures ``rag_chunks.vector_index`` for 768 dimensions,
    while the memory index uses 3072. This RAG-only helper explicitly requests
    the RAG index dimension and retrieval task type.

    When caching is enabled, only float vectors are stored in Redis — never
    document text. Cache misses/failures fall back to the provider.
    """
    from app.services.rag_config import settings as rag_settings
    from app.services.rag_embedding_cache import (
        get_cached_embedding,
        set_cached_embedding,
    )

    values = [str(text) for text in texts]
    if not values or any(not text.strip() for text in values):
        raise ValueError("RAG embedding input cannot be empty")
    if dimensions < 1 or not 1 <= batch_size <= 100:
        raise ValueError("Invalid RAG embedding dimensions or batch size")
    if task_type not in {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"}:
        raise ValueError("Unsupported RAG embedding task type")

    version = embedding_version or rag_settings.embedding_version
    cache_enabled = rag_settings.embedding_cache if use_cache is None else bool(use_cache)

    output: list[list[float] | None] = [None] * len(values)
    missing_indexes: list[int] = []
    for index, text in enumerate(values):
        if cache_enabled:
            cached = get_cached_embedding(
                text=text,
                model=model,
                dimensions=dimensions,
                task_type=task_type,
                embedding_version=version,
                chunking_version=chunking_version,
            )
            if cached is not None:
                output[index] = cached
                continue
        missing_indexes.append(index)

    for batch_start in range(0, len(missing_indexes), batch_size):
        batch_indexes = missing_indexes[batch_start : batch_start + batch_size]
        batch = [values[index] for index in batch_indexes]
        for attempt in range(3):
            try:
                response = rag_client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=dimensions,
                    ),
                )
                embeddings = [row.values for row in (response.embeddings or [])]
                if len(embeddings) != len(batch):
                    raise ValueError("Embedding provider returned an incomplete batch")
                for local_index, vector in enumerate(embeddings):
                    if len(vector) != dimensions or not all(math.isfinite(float(n)) for n in vector):
                        raise ValueError("Embedding provider returned an invalid vector")
                    floats = [float(n) for n in vector]
                    absolute_index = batch_indexes[local_index]
                    output[absolute_index] = floats
                    if cache_enabled:
                        set_cached_embedding(
                            text=values[absolute_index],
                            model=model,
                            dimensions=dimensions,
                            task_type=task_type,
                            embedding_version=version,
                            vector=floats,
                            chunking_version=chunking_version,
                        )
                break
            except ValueError:
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                try:
                    numeric_status = int(status) if status is not None else None
                except (TypeError, ValueError):
                    numeric_status = None
                if attempt == 2 or (numeric_status is not None and numeric_status not in {408, 429, 500, 502, 503, 504}):
                    # Do not expose provider error strings, which can include request details.
                    raise RuntimeError(f"RAG embedding provider failed ({type(exc).__name__})") from exc
                time.sleep(0.2 * (2 ** attempt))

    if any(item is None for item in output):
        raise RuntimeError("RAG embedding provider failed (incomplete cache fill)")
    return output  # type: ignore[return-value]


def create_rag_embedding(text, *, task_type, dimensions=768, model=EMBEDDING_MODEL):
    """Single-text wrapper used by query retrieval."""
    return create_rag_embeddings([text], task_type=task_type, dimensions=dimensions, model=model)[0]


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    try:

        if not text or not text.strip():

            raise ValueError(
                "Text cannot be empty"
            )


        response = client.models.embed_content(

            model=EMBEDDING_MODEL,

            contents=text

        )


        embedding = response.embeddings[0].values


        print(
            f"[Embedding generated] "
            f"Dimensions: {len(embedding)}"
        )


        return embedding


    except Exception as e:
        # Provider exceptions can contain request metadata; don't emit headers,
        # endpoint details, or credentials into application logs.
        print(f"[Embedding error]: {type(e).__name__}")

        raise
