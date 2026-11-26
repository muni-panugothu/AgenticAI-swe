"""In-process Mongo-like collections for durable security state in tests.

Production uses the shared Mongo ``database`` from memory_service.
This module provides a concurrency-safe fake with ``find_one_and_update``.
"""

from __future__ import annotations

import copy
import threading
from typing import Any


def _match(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict):
            if "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif "$gt" in expected:
                if actual is None or not (actual > expected["$gt"]):
                    return False
            elif "$ne" in expected:
                if actual == expected["$ne"]:
                    return False
            else:
                return False
        else:
            if actual != expected:
                return False
    return True


class InMemoryCollection:
    """Minimal collection supporting insert/find/find_one_and_update."""

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def insert_one(self, doc: dict[str, Any]) -> Any:
        with self._lock:
            self._docs.append(copy.deepcopy(doc))

        class _Result:
            inserted_id = doc.get("approval_id") or doc.get("grant_id")

        return _Result()

    def find(self, query: dict[str, Any] | None = None):
        query = query or {}
        with self._lock:
            return [copy.deepcopy(d) for d in self._docs if _match(d, query)]

    def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            for doc in self._docs:
                if _match(doc, query):
                    return copy.deepcopy(doc)
        return None

    def find_one_and_update(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
        return_document: Any = True,
    ) -> dict[str, Any] | None:
        with self._lock:
            for i, doc in enumerate(self._docs):
                if not _match(doc, query):
                    continue
                new_doc = copy.deepcopy(doc)
                if "$set" in update:
                    new_doc.update(update["$set"])
                self._docs[i] = new_doc
                return copy.deepcopy(new_doc)
        return None

    def create_index(self, *args, **kwargs) -> None:
        return None

    def delete_many(self, query: dict[str, Any] | None = None) -> None:
        query = query or {}
        with self._lock:
            self._docs = [d for d in self._docs if not _match(d, query)]
