"""Token counting for RAG context budgets (tiktoken, pinned encoding)."""

from __future__ import annotations

from functools import lru_cache

_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text))


def encoding_name() -> str:
    return _ENCODING_NAME
