from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from datetime import date, datetime, time
import math
from types import MappingProxyType
from typing import Any


def freeze_deterministic_json_value(value: Any) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("deterministic_json_invalid")
        return value
    if isinstance(value, Mapping):
        normalized_items: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("deterministic_json_invalid")
            normalized_items[key] = freeze_deterministic_json_value(item)
        return MappingProxyType(normalized_items)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_deterministic_json_value(item) for item in value)
    if isinstance(value, (bytes, bytearray, datetime, date, time)):
        raise ValueError("deterministic_json_invalid")
    if is_dataclass(value):
        raise ValueError("deterministic_json_invalid")
    raise ValueError("deterministic_json_invalid")
