from typing import Any

import aiohttp
import pytest

from app.adapters.ai_client import ChatResult, Message, UniversalChatClient


class _FakeResponse:
    def __init__(self, json_data: dict[str, Any]) -> None:
        self._json_data = json_data

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return self._json_data


class _FakeResponseCtx:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeSession:
    def __init__(self, response_ctx: Any = None, raise_error: Exception | None = None) -> None:
        self._response_ctx = response_ctx
        self._raise_error = raise_error
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, json: dict, headers: dict, timeout: Any) -> Any:
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._raise_error is not None:
            raise self._raise_error
        return self._response_ctx


def _messages() -> list[Message]:
    return [{"role": "user", "content": "хочу снять квартиру"}]


@pytest.mark.asyncio
async def test_chat_success_parses_content_and_usage() -> None:
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
    response = _FakeResponse({"unexpected": "shape"})
    session = _FakeSession(response_ctx=_FakeResponseCtx(response))
    client = UniversalChatClient(session)  # type: ignore[arg-type]

    result = await client.chat(_messages())

    assert result.content is None


@pytest.mark.asyncio
async def test_chat_connection_error_returns_empty_chat_result() -> None:
    session = _FakeSession(raise_error=aiohttp.ClientConnectionError("boom"))
    client = UniversalChatClient(session)  # type: ignore[arg-type]

    result = await client.chat(_messages())

    assert result == ChatResult(
        content=None, prompt_tokens=0, completion_tokens=0, total_tokens=0
    )


@pytest.mark.asyncio
async def test_chat_unexpected_error_returns_empty_chat_result() -> None:
    session = _FakeSession(raise_error=ValueError("something else broke"))
    client = UniversalChatClient(session)  # type: ignore[arg-type]

    result = await client.chat(_messages())

    assert result.content is None
