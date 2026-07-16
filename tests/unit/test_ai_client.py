from typing import Any

import aiohttp
import pytest

from app.adapters.ai_client import ChatResult, Message, UniversalChatClient


class _FakeResponse:
    """Stand-in for an aiohttp response, exposing only what UniversalChatClient reads"""

    def __init__(self, json_data: dict[str, Any]) -> None:
        """Store the JSON body this fake response will return"""
        self._json_data = json_data

    def raise_for_status(self) -> None:
        """No-op: the fake response is always considered a success"""
        return None

    async def json(self) -> dict[str, Any]:
        """Return the canned JSON body"""
        return self._json_data


class _FakeResponseCtx:
    """Async context manager wrapper around a _FakeResponse, mirroring session.post(...)"""

    def __init__(self, response: _FakeResponse) -> None:
        """Store the response to hand back on __aenter__"""
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        """Return the wrapped fake response"""
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        """Do not suppress any exception raised inside the `async with` block"""
        return False


class _FakeSession:
    """Stand-in for aiohttp.ClientSession: records calls and returns a canned response or raises"""

    def __init__(
        self, response_ctx: Any = None, raise_error: Exception | None = None
    ) -> None:
        """Configure what post() should return or raise, and start the call log empty"""
        self._response_ctx = response_ctx
        self._raise_error = raise_error
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, json: dict, headers: dict, timeout: Any) -> Any:
        """Record the call and either raise the configured error or return the response context"""
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._raise_error is not None:
            raise self._raise_error
        return self._response_ctx


def _messages() -> list[Message]:
    """Build a minimal one-message chat payload for tests"""
    return [{"role": "user", "content": "хочу снять квартиру"}]


@pytest.mark.asyncio
async def test_chat_success_parses_content_and_usage() -> None:
    """A well-formed response yields content plus token usage, and forwards json_mode/temperature"""
    response = _FakeResponse(
        {
            "choices": [{"message": {"content": '{"category": "rent"}'}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )
    session = _FakeSession(response_ctx=_FakeResponseCtx(response))
    client = UniversalChatClient(session)  # type: ignore[arg-type]

    result = await client.chat(_messages(), json_mode=True, temperature=0.2)

    assert result == ChatResult(
        content='{"category": "rent"}',
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    assert session.calls[0]["json"]["response_format"] == {"type": "json_object"}
    assert session.calls[0]["json"]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_chat_missing_usage_defaults_to_zero() -> None:
    """A response without a usage block still returns content, with token counts at zero"""
    response = _FakeResponse({"choices": [{"message": {"content": "ok"}}]})
    session = _FakeSession(response_ctx=_FakeResponseCtx(response))
    client = UniversalChatClient(session)  # type: ignore[arg-type]

    result = await client.chat(_messages())

    assert result.content == "ok"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_chat_malformed_response_shape_returns_none_content() -> None:
    """A response missing the expected choices/message/content shape yields content=None, not a crash"""
    response = _FakeResponse({"unexpected": "shape"})
    session = _FakeSession(response_ctx=_FakeResponseCtx(response))
    client = UniversalChatClient(session)  # type: ignore[arg-type]

    result = await client.chat(_messages())

    assert result.content is None


@pytest.mark.asyncio
async def test_chat_connection_error_returns_empty_chat_result() -> None:
    """A network-level ClientConnectionError is swallowed into an empty ChatResult, not raised"""
    session = _FakeSession(raise_error=aiohttp.ClientConnectionError("boom"))
    client = UniversalChatClient(session)  # type: ignore[arg-type]

    result = await client.chat(_messages())

    assert result == ChatResult(
        content=None, prompt_tokens=0, completion_tokens=0, total_tokens=0
    )


@pytest.mark.asyncio
async def test_chat_unexpected_error_returns_empty_chat_result() -> None:
    """Any other unexpected exception from the transport is also swallowed, not propagated"""
    session = _FakeSession(raise_error=ValueError("something else broke"))
    client = UniversalChatClient(session)  # type: ignore[arg-type]

    result = await client.chat(_messages())

    assert result.content is None
