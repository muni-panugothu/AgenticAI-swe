"""Request-scoped cooperative cancellation (Topic 42c).

Isolated to one chat request. Cancelling one request never affects another.
Does not forcibly kill OS threads — workers must poll ``is_cancelled`` /
``check()`` at safe boundaries.
"""

from __future__ import annotations

import threading
from typing import Any


class RequestCancelled(Exception):
    """Raised when cooperative cancellation is observed at a safe boundary."""

    def __init__(self, reason: str = "client_disconnect") -> None:
        self.reason = str(reason or "client_disconnect")
        super().__init__(f"request_cancelled:{self.reason}")


class CancellationToken:
    """Thread-safe cancellation flag for one request."""

    __slots__ = ("_event", "_reason", "_request_id", "_user_id")

    def __init__(
        self,
        *,
        request_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self._event = threading.Event()
        self._reason = "client_disconnect"
        self._request_id = request_id
        self._user_id = user_id

    @property
    def request_id(self) -> str | None:
        return self._request_id

    @property
    def user_id(self) -> str | None:
        return self._user_id

    def cancel(self, reason: str = "client_disconnect") -> None:
        self._reason = str(reason or "client_disconnect")
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def check(self) -> None:
        if self._event.is_set():
            raise RequestCancelled(self._reason)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancelled or timeout. Returns True if cancelled."""
        return self._event.wait(timeout)


def get_cancellation(obj: Any) -> CancellationToken | None:
    if obj is None:
        return None
    if isinstance(obj, CancellationToken):
        return obj
    return None
