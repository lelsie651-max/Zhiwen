from __future__ import annotations

import hashlib


def build_document_block_anchor_hash(
    *,
    detected_format: str,
    location_key: str,
    raw_text: str,
) -> str:
    if (
        not isinstance(detected_format, str)
        or not isinstance(location_key, str)
        or not isinstance(raw_text, str)
    ):
        raise ValueError("document_block_anchor_hash_invalid_input")
    return hashlib.sha256(
        f"{detected_format}|{location_key}|{raw_text}".encode("utf-8")
    ).hexdigest()
