"""Structured operation tracing for the memory intelligence layer."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger("memory_intelligence")


def log_memory_operation(
    operation: str,
    user_id: str,
    *,
    success: bool,
    memory_id: str | None = None,
    operation_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "user_id": user_id,
        "success": success,
        "memory_id": memory_id,
        "operation_id": operation_id,
        "details": details or {},
    }
    message = json.dumps(payload, default=str, ensure_ascii=False)
    if success:
        logger.info(message)
    else:
        logger.warning(message)
    # Keep stdout visibility for the existing chatbot console workflow.
    print(f"[MemoryTrace] {message}")
