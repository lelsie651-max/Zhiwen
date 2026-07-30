import re


HANDLE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,31}$")
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


def normalize_text(value: str) -> str:
    return value.strip()


def normalize_handle(value: str) -> str:
    return normalize_text(value).lower()


def normalize_slug(value: str) -> str:
    return normalize_text(value).lower()


def normalize_optional_email(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = normalize_text(value).lower()
    return normalized or None


def validate_handle(value: str) -> str:
    normalized = normalize_handle(value)
    if not HANDLE_PATTERN.fullmatch(normalized):
        raise ValueError(
            "handle must be 3-32 chars and use lowercase letters, digits, _ or -."
        )
    return normalized


def validate_slug(value: str) -> str:
    normalized = normalize_slug(value)
    if not SLUG_PATTERN.fullmatch(normalized):
        raise ValueError(
            "slug must be 3-64 chars and use lowercase letters, digits or -."
        )
    return normalized
