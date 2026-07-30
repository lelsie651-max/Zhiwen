"""LLM client foundation.

A thin, provider-agnostic wrapper over any OpenAI-compatible ``/chat/completions``
endpoint. It works unchanged against:

- DeepSeek official API (``https://api.deepseek.com/v1``, model ``deepseek-chat``)
- Alibaba Cloud Bailian / DashScope compatible mode
  (``https://dashscope.aliyuncs.com/compatible-mode/v1``, model ``deepseek-v3``)
- Any other OpenAI-compatible host

The module intentionally does not depend on a vendor SDK: it speaks the raw HTTP
contract with ``httpx`` so the same code path serves every provider and stays easy
to mock in tests. Higher layers (Agent 1 fact/schema proposal) build on
``LLMClient`` and never touch transport details.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import httpx

from app.core.config import Settings, get_settings


class LLMError(Exception):
    """Base class for all LLM client failures."""


class LLMConfigurationError(LLMError):
    """Raised when the client is used without required configuration (e.g. API key)."""


class LLMTransportError(LLMError):
    """Raised when the upstream endpoint cannot be reached or returns an error status."""


class _RetryableTransportError(LLMTransportError):
    """Internal marker for transient failures worth retrying (5xx / 429 / network)."""


class LLMResponseError(LLMError):
    """Raised when the upstream reply is malformed or cannot be parsed as expected."""


@dataclass(frozen=True)
class LLMMessage:
    """A single chat message. ``role`` is one of system/user/assistant."""

    role: str
    content: str

    def as_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _messages_to_payload(messages: Sequence[LLMMessage]) -> list[dict[str, str]]:
    return [message.as_payload() for message in messages]


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM reply.

    Handles the common shapes models return:
    - a bare JSON document,
    - JSON wrapped in a ```json ... ``` fenced block,
    - JSON preceded/followed by prose.

    Raises :class:`LLMResponseError` when no valid JSON can be recovered.
    """

    if text is None:
        raise LLMResponseError("empty LLM response")

    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    for match in _JSON_FENCE_PATTERN.finditer(text):
        fenced = match.group(1).strip()
        if fenced:
            candidates.append(fenced)

    # Fall back to the first balanced {...} or [...] span in the raw text.
    span = _first_json_span(text)
    if span is not None:
        candidates.append(span)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise LLMResponseError("LLM response did not contain valid JSON")


def _first_json_span(text: str) -> str | None:
    start_index: int | None = None
    opener: str | None = None
    closer: str | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start_index is None:
            if char in "{[":
                start_index = index
                opener = char
                closer = "}" if char == "{" else "]"
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]

    return None


class LLMClient(Protocol):
    """Structural interface every LLM backend must satisfy."""

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        """Return the assistant message content for ``messages``."""
        ...


class OpenAICompatibleClient:
    """LLM client speaking the OpenAI ``/chat/completions`` HTTP contract."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        default_temperature: float = 0.2,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("LLM API key is not configured")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._default_temperature = default_temperature
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _messages_to_payload(messages),
            "temperature": (
                temperature if temperature is not None else self._default_temperature
            ),
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds, transport=self._transport
                ) as client:
                    response = await client.post(url, json=payload, headers=headers)
                if response.status_code >= 500 or response.status_code == 429:
                    raise _RetryableTransportError(
                        f"LLM endpoint returned retryable status {response.status_code}"
                    )
                if response.status_code >= 400:
                    # Client errors (auth, bad request) will not succeed on retry.
                    raise LLMTransportError(
                        f"LLM endpoint returned status {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                return _parse_chat_content(response.json())
            except (httpx.HTTPError, _RetryableTransportError) as error:
                last_error = error
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(_backoff_seconds(attempt))

        raise LLMTransportError(
            f"LLM request failed after {self._max_retries + 1} attempt(s): {last_error}"
        ) from last_error


def _backoff_seconds(attempt: int) -> float:
    return min(2.0 ** attempt, 8.0)


def _parse_chat_content(body: Any) -> str:
    if not isinstance(body, dict):
        raise LLMResponseError("LLM response body was not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("LLM response contained no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LLMResponseError("LLM response choice had no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise LLMResponseError("LLM response message had no text content")
    return content


class MockLLMClient:
    """Deterministic in-memory client for tests and offline development.

    Provide either a fixed list of ``responses`` (returned in order, then the
    last one repeats) or a ``handler`` callable receiving the message list.
    Every call is recorded in :attr:`calls` for assertions.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        handler: Any | None = None,
    ) -> None:
        if responses is None and handler is None:
            raise ValueError("MockLLMClient needs either responses or a handler")
        self._responses = list(responses) if responses is not None else None
        self._handler = handler
        self._cursor = 0
        self.calls: list[list[LLMMessage]] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        recorded = list(messages)
        self.calls.append(recorded)
        if self._handler is not None:
            result = self._handler(recorded)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        assert self._responses is not None  # narrowed by __init__
        if not self._responses:
            raise LLMResponseError("MockLLMClient has no configured responses")
        index = min(self._cursor, len(self._responses) - 1)
        self._cursor += 1
        return self._responses[index]


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Construct the configured LLM client.

    Raises :class:`LLMConfigurationError` when no API key is set, so callers can
    fail fast (or inject a :class:`MockLLMClient` in tests).
    """

    resolved = settings or get_settings()
    if not resolved.llm_configured:
        raise LLMConfigurationError(
            "LLM is not configured; set LLM_API_KEY (and optionally LLM_BASE_URL/LLM_MODEL)"
        )
    return OpenAICompatibleClient(
        api_key=resolved.llm_api_key,
        base_url=resolved.llm_base_url,
        model=resolved.llm_model,
        default_temperature=resolved.llm_temperature,
        timeout_seconds=resolved.llm_timeout_seconds,
        max_retries=resolved.llm_max_retries,
    )
