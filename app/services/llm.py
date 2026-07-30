"""LLM client foundation with a traceable, strict response contract.

A thin, provider-agnostic wrapper over any OpenAI-compatible ``/chat/completions``
endpoint. It works unchanged against:

- DeepSeek official API (``https://api.deepseek.com``, model ``deepseek-v4-flash``)
- Alibaba Cloud Bailian / DashScope compatible mode
  (``https://dashscope.aliyuncs.com/compatible-mode/v1``, model ``deepseek-v3``)
- Any other OpenAI-compatible host

Design goals:

- **Traceability.** ``complete()`` returns a rich :class:`LLMCompletion` carrying
  response id, actual model, ``finish_reason``, token usage (including cache and
  reasoning tokens) and the real attempt count. Higher layers never see a bare
  string, so every AI-derived fact can answer "which model, which call, how many
  tokens, was it truncated".
- **Safety.** API keys are handled as secrets and never enter error messages; raw
  upstream response bodies are never copied into exceptions, logs, or persisted
  fields.
- **Strictness.** ``finish_reason`` is honoured (truncated / filtered replies are
  rejected, not silently accepted), and authoritative agent output must go through
  :func:`parse_strict_json_object`, which refuses fenced blocks, surrounding prose,
  and non-object top levels.

The module intentionally does not depend on a vendor SDK: it speaks the raw HTTP
contract with ``httpx`` so the same code path serves every provider and stays easy
to mock in tests.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, Sequence

import httpx

from app.core.config import Settings, get_settings


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class LLMError(Exception):
    """Base class for all LLM client failures."""


class LLMConfigurationError(LLMError):
    """Raised when the client is used without required configuration (e.g. API key)."""


class LLMRequestError(LLMError):
    """Raised when a request is rejected locally before any HTTP call is made."""


class LLMResponseError(LLMError):
    """Raised when the upstream reply is malformed, unsafe, or otherwise unusable."""


class LLMIncompleteResponseError(LLMResponseError):
    """Raised when the model stopped early (``finish_reason == "length"``)."""


class LLMTransportError(LLMError):
    """Raised when the endpoint cannot be reached or returns an error status.

    Only carries safe metadata (status code, a stable error code, whether the
    failure is retryable). It never contains the API key, request headers, prompt
    content, or the raw upstream response body.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        error_code: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.error_code = error_code


class _RetrySignal(Exception):
    """Internal control-flow marker wrapping the public error to surface if exhausted."""

    def __init__(self, public_error: LLMError) -> None:
        super().__init__(str(public_error))
        self.public_error = public_error


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_FINISH_STOP = "stop"


@dataclass(frozen=True)
class LLMMessage:
    """A single chat message. ``role`` is one of system/user/assistant."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ALLOWED_ROLES:
            raise LLMRequestError(
                "message role must be one of system, user, assistant"
            )
        if not isinstance(self.content, str) or not self.content.strip():
            raise LLMRequestError("message content must be a non-blank string")

    def as_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMUsage:
    """Token accounting for one completion. Any field may be ``None`` if the
    provider does not report it."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_cache_hit_tokens: int | None
    prompt_cache_miss_tokens: int | None
    reasoning_tokens: int | None

    @staticmethod
    def empty() -> "LLMUsage":
        return LLMUsage(None, None, None, None, None, None)


@dataclass(frozen=True)
class LLMCompletion:
    """The full, traceable result of a single ``complete()`` call."""

    content: str
    provider: str
    model: str
    response_id: str | None
    system_fingerprint: str | None
    finish_reason: str
    usage: LLMUsage
    attempt_count: int


# --------------------------------------------------------------------------- #
# JSON parsing helpers
# --------------------------------------------------------------------------- #


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Best-effort, **non-authoritative** extraction of JSON from an LLM reply.

    Handles bare JSON, ```json``` fenced blocks, and JSON surrounded by prose.
    Useful for debugging and lenient tooling only.

    WARNING: Do NOT use this to parse authoritative agent output — it will happily
    accept JSON embedded inside adversarial prose (prompt-injection risk). Agent
    structured output MUST use :func:`parse_strict_json_object` instead.

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

    span = _first_json_span(text)
    if span is not None:
        candidates.append(span)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise LLMResponseError("LLM response did not contain valid JSON")


def parse_strict_json_object(text: str) -> dict[str, Any]:
    """Authoritative parser for agent structured output.

    The *entire* trimmed content must be a single JSON document whose top level is
    an object. Fenced code blocks, leading/trailing prose, and non-object top
    levels (arrays, strings, numbers, booleans, null) are all rejected. This is the
    only parser agents may use for their real output; :func:`extract_json` is for
    debugging aids, never authoritative results.

    Raises :class:`LLMResponseError` on any deviation.
    """

    if not isinstance(text, str):
        raise LLMResponseError("strict JSON input must be a string")
    trimmed = text.strip()
    if not trimmed:
        raise LLMResponseError("strict JSON content was empty")
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError as error:
        raise LLMResponseError("content was not a single valid JSON document") from error
    if not isinstance(parsed, dict):
        raise LLMResponseError("strict JSON top level must be an object")
    return parsed


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


# --------------------------------------------------------------------------- #
# Client protocol
# --------------------------------------------------------------------------- #


class LLMClient(Protocol):
    """Structural interface every LLM backend must satisfy."""

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> LLMCompletion:
        """Return a traceable :class:`LLMCompletion` for ``messages``."""
        ...


def _validate_request(
    messages: Sequence[LLMMessage],
    *,
    temperature: float | None,
    max_tokens: int | None,
    default_temperature: float,
    default_max_tokens: int,
    response_format_json: bool,
) -> tuple[list[LLMMessage], float, int]:
    resolved_messages = list(messages)
    if not resolved_messages:
        raise LLMRequestError("at least one message is required")
    for message in resolved_messages:
        if not isinstance(message, LLMMessage):
            raise LLMRequestError("messages must be LLMMessage instances")

    if temperature is None:
        resolved_temperature = default_temperature
    else:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise LLMRequestError("temperature must be a number")
        if not math.isfinite(float(temperature)) or not 0.0 <= float(temperature) <= 2.0:
            raise LLMRequestError("temperature must be a finite value within [0, 2]")
        resolved_temperature = float(temperature)

    if max_tokens is None:
        resolved_max_tokens = default_max_tokens
    else:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise LLMRequestError("max_tokens must be an integer")
        if max_tokens <= 0:
            raise LLMRequestError("max_tokens must be a positive integer")
        resolved_max_tokens = max_tokens

    if response_format_json:
        has_json_hint = any(
            "json" in message.content.lower() for message in resolved_messages
        )
        if not has_json_hint:
            raise LLMRequestError(
                "JSON response format requires the word 'json' in the prompt"
            )

    return resolved_messages, resolved_temperature, resolved_max_tokens


# --------------------------------------------------------------------------- #
# Real client
# --------------------------------------------------------------------------- #


class OpenAICompatibleClient:
    """LLM client speaking the OpenAI ``/chat/completions`` HTTP contract."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider: str = "openai-compatible",
        default_temperature: float = 0.2,
        default_max_tokens: int = 8192,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("LLM API key is not configured")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._provider = provider
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> LLMCompletion:
        resolved_messages, resolved_temperature, resolved_max_tokens = _validate_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            default_temperature=self._default_temperature,
            default_max_tokens=self._default_max_tokens,
            response_format_json=response_format_json,
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.as_payload() for message in resolved_messages],
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: LLMError | None = None
        # One AsyncClient for the whole call — retries reuse the same connection pool.
        async with httpx.AsyncClient(
            timeout=self._timeout_seconds, transport=self._transport
        ) as client:
            for attempt in range(self._max_retries + 1):
                attempt_count = attempt + 1
                try:
                    response = await client.post(url, json=payload, headers=headers)
                except httpx.HTTPError as error:  # network-level failure
                    last_error = LLMTransportError(
                        "network error contacting LLM endpoint",
                        status_code=None,
                        retryable=True,
                        error_code="network_error",
                    )
                    _ = error  # deliberately not surfaced (may carry sensitive detail)
                    if attempt >= self._max_retries:
                        break
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue

                if response.status_code != 200:
                    error = _map_status_error(response.status_code)
                    if error.retryable and attempt < self._max_retries:
                        last_error = error
                        await asyncio.sleep(_backoff_seconds(attempt))
                        continue
                    raise error

                try:
                    return _build_completion(
                        response,
                        provider=self._provider,
                        fallback_model=self._model,
                        attempt_count=attempt_count,
                        response_format_json=response_format_json,
                    )
                except _RetrySignal as signal:
                    last_error = signal.public_error
                    if attempt >= self._max_retries:
                        break
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue

        raise last_error or LLMTransportError(
            "LLM request failed", retryable=True, error_code="network_error"
        )


def _backoff_seconds(attempt: int) -> float:
    return min(2.0 ** attempt, 8.0)


def _map_status_error(status_code: int) -> LLMTransportError:
    if status_code in (401, 403):
        return LLMTransportError(
            f"LLM endpoint returned status {status_code}",
            status_code=status_code,
            retryable=False,
            error_code="authentication_failed",
        )
    if status_code == 408:
        return LLMTransportError(
            "LLM endpoint request timed out",
            status_code=status_code,
            retryable=True,
            error_code="request_timeout",
        )
    if status_code == 429:
        return LLMTransportError(
            "LLM endpoint rate limited the request",
            status_code=status_code,
            retryable=True,
            error_code="rate_limited",
        )
    if status_code >= 500:
        return LLMTransportError(
            f"LLM endpoint returned status {status_code}",
            status_code=status_code,
            retryable=True,
            error_code="upstream_unavailable",
        )
    return LLMTransportError(
        f"LLM endpoint returned status {status_code}",
        status_code=status_code,
        retryable=False,
        error_code="client_error",
    )


def _build_completion(
    response: httpx.Response,
    *,
    provider: str,
    fallback_model: str,
    attempt_count: int,
    response_format_json: bool,
) -> LLMCompletion:
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError) as error:
        raise LLMResponseError("LLM response body was not valid JSON") from error

    if not isinstance(body, dict):
        raise LLMResponseError("LLM response body was not a JSON object")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("LLM response contained no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseError("LLM response choice was malformed")

    finish_reason = first.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise LLMResponseError("LLM response had no finish_reason")

    if finish_reason == "length":
        raise LLMIncompleteResponseError(
            "LLM response was truncated (finish_reason=length)"
        )
    if finish_reason == "content_filter":
        raise LLMResponseError("LLM response was blocked by the content filter")
    if finish_reason == "tool_calls":
        raise LLMResponseError("LLM returned tool_calls, which are not supported")
    if finish_reason == "insufficient_system_resource":
        raise _RetrySignal(
            LLMTransportError(
                "LLM endpoint reported insufficient system resource",
                status_code=None,
                retryable=True,
                error_code="upstream_unavailable",
            )
        )
    if finish_reason != _FINISH_STOP:
        raise LLMResponseError(f"unexpected finish_reason: {finish_reason}")

    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMResponseError("LLM response choice had no message")
    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        empty_error = LLMResponseError("LLM response message had empty content")
        # DeepSeek's JSON mode can occasionally return empty content; only that
        # case is worth a bounded retry. In plain mode an empty reply is terminal.
        if response_format_json:
            raise _RetrySignal(empty_error)
        raise empty_error

    response_id = body.get("id")
    if response_id is not None and not isinstance(response_id, str):
        raise LLMResponseError("LLM response id had an invalid type")
    system_fingerprint = body.get("system_fingerprint")
    if system_fingerprint is not None and not isinstance(system_fingerprint, str):
        raise LLMResponseError("LLM system_fingerprint had an invalid type")
    model = body.get("model")
    resolved_model = model if isinstance(model, str) and model else fallback_model

    usage = _parse_usage(body.get("usage"))

    return LLMCompletion(
        content=content,
        provider=provider,
        model=resolved_model,
        response_id=response_id,
        system_fingerprint=system_fingerprint,
        finish_reason=finish_reason,
        usage=usage,
        attempt_count=attempt_count,
    )


def _usage_int(raw: dict[str, Any], name: str) -> int | None:
    if name not in raw:
        return None
    value = raw[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMResponseError(f"usage.{name} had an invalid type")
    return value


def _parse_usage(raw: Any) -> LLMUsage:
    if raw is None:
        return LLMUsage.empty()
    if not isinstance(raw, dict):
        raise LLMResponseError("usage field was not an object")

    reasoning_tokens: int | None = None
    details = raw.get("completion_tokens_details")
    if isinstance(details, dict) and "reasoning_tokens" in details:
        reasoning_tokens = _usage_int(details, "reasoning_tokens")
    elif "reasoning_tokens" in raw:
        reasoning_tokens = _usage_int(raw, "reasoning_tokens")

    return LLMUsage(
        prompt_tokens=_usage_int(raw, "prompt_tokens"),
        completion_tokens=_usage_int(raw, "completion_tokens"),
        total_tokens=_usage_int(raw, "total_tokens"),
        prompt_cache_hit_tokens=_usage_int(raw, "prompt_cache_hit_tokens"),
        prompt_cache_miss_tokens=_usage_int(raw, "prompt_cache_miss_tokens"),
        reasoning_tokens=reasoning_tokens,
    )


# --------------------------------------------------------------------------- #
# Mock client
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MockCall:
    """One recorded call to :class:`MockLLMClient`."""

    messages: tuple[LLMMessage, ...]
    temperature: float | None
    max_tokens: int | None
    response_format_json: bool


def make_stub_completion(
    content: str,
    *,
    provider: str = "mock",
    model: str = "mock-model",
    response_id: str | None = "mock-response",
    system_fingerprint: str | None = None,
    finish_reason: str = "stop",
    usage: LLMUsage | None = None,
    attempt_count: int = 1,
) -> LLMCompletion:
    """Wrap a string into a deterministic :class:`LLMCompletion` for tests."""

    return LLMCompletion(
        content=content,
        provider=provider,
        model=model,
        response_id=response_id,
        system_fingerprint=system_fingerprint,
        finish_reason=finish_reason,
        usage=usage or LLMUsage.empty(),
        attempt_count=attempt_count,
    )


MockHandler = Callable[
    [list[LLMMessage]],
    "str | LLMCompletion | Awaitable[str | LLMCompletion]",
]


class MockLLMClient:
    """Deterministic in-memory client for tests and offline development.

    Provide either a fixed list of ``responses`` (each a ``str`` — auto-wrapped
    into a stub completion — or an :class:`LLMCompletion`) or a ``handler``
    callable receiving the message list. Unlike a lenient mock, responses do NOT
    repeat by default: once the list is exhausted a call raises, so tests catch
    unexpected extra LLM calls. Pass ``repeat_last=True`` to opt into repetition.
    Every call is recorded in :attr:`calls`.
    """

    def __init__(
        self,
        responses: Sequence[str | LLMCompletion] | None = None,
        *,
        handler: MockHandler | None = None,
        repeat_last: bool = False,
    ) -> None:
        if responses is None and handler is None:
            raise ValueError("MockLLMClient needs either responses or a handler")
        if responses is not None and handler is not None:
            raise ValueError("provide either responses or a handler, not both")
        self._responses = list(responses) if responses is not None else None
        self._handler = handler
        self._repeat_last = repeat_last
        self._cursor = 0
        self.calls: list[MockCall] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> LLMCompletion:
        recorded = list(messages)
        self.calls.append(
            MockCall(
                messages=tuple(recorded),
                temperature=temperature,
                max_tokens=max_tokens,
                response_format_json=response_format_json,
            )
        )

        if self._handler is not None:
            result = self._handler(recorded)
            if asyncio.iscoroutine(result):
                result = await result
            return _coerce_completion(result)

        assert self._responses is not None  # narrowed by __init__
        if self._cursor >= len(self._responses):
            if self._repeat_last and self._responses:
                return _coerce_completion(self._responses[-1])
            raise LLMResponseError("MockLLMClient exhausted its configured responses")
        item = self._responses[self._cursor]
        self._cursor += 1
        return _coerce_completion(item)


def _coerce_completion(value: Any) -> LLMCompletion:
    if isinstance(value, LLMCompletion):
        return value
    if isinstance(value, str):
        return make_stub_completion(value)
    raise LLMResponseError("mock response must be a str or LLMCompletion")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


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
        api_key=resolved.llm_api_key.get_secret_value(),
        base_url=resolved.llm_base_url,
        model=resolved.llm_model,
        provider=resolved.llm_provider,
        default_temperature=resolved.llm_temperature,
        default_max_tokens=resolved.llm_max_output_tokens,
        timeout_seconds=resolved.llm_timeout_seconds,
        max_retries=resolved.llm_max_retries,
    )
