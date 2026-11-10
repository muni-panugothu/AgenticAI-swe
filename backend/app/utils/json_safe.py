"""Recursive JSON-safe conversion for tool results and API payloads.

Converts MongoDB/BSON and Python types that the stdlib json module
cannot serialize, without using blanket str() for unknown objects.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID


def _object_id_type():
    try:
        from bson import ObjectId
        return ObjectId
    except Exception:
        return None


def to_json_safe(value: Any) -> Any:
    """Recursively convert a value into JSON-serializable primitives.

    Supported conversions:
    - datetime / date → ISO 8601 string (timezone preserved when present)
    - ObjectId → str
    - UUID → str
    - Decimal → float when finite, else str
    - Enum → value
    - Pydantic models → model_dump then recurse
    - dict / list / tuple / set → recurse
    - bytes → utf-8 string with replacement
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    object_id_type = _object_id_type()
    if object_id_type is not None and isinstance(value, object_id_type):
        return str(value)

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Decimal):
        as_float = float(value)
        if as_float == value:
            return as_float
        return str(value)

    if isinstance(value, Enum):
        return to_json_safe(value.value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    # Pydantic v2 models
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_json_safe(model_dump())

    # Pydantic v1 / objects with dict()
    as_dict = getattr(value, "dict", None)
    if callable(as_dict) and not isinstance(value, type):
        try:
            return to_json_safe(as_dict())
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def dumps_json_safe(value: Any, *, ensure_ascii: bool = False) -> str:
    """Serialize a value to JSON after recursive sanitization."""
    return json.dumps(to_json_safe(value), ensure_ascii=ensure_ascii)


def is_json_serializable(value: Any) -> bool:
    try:
        dumps_json_safe(value)
        return True
    except (TypeError, ValueError):
        return False
