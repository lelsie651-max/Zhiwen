from __future__ import annotations

import math
from typing import Any

from app.schemas.fact import _normalize_optional_text


def validate_fact_value_language_code(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("fact_value_language_code_invalid")
    normalized = _normalize_optional_text(
        value,
        field_name="language_code",
        max_length=32,
    )
    if normalized != value:
        raise ValueError("fact_value_language_code_invalid")
    return normalized


def validate_fact_value_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("fact_value_confidence_invalid")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("fact_value_confidence_invalid")
    if numeric < 0 or numeric > 1:
        raise ValueError("fact_value_confidence_invalid")
    return numeric
