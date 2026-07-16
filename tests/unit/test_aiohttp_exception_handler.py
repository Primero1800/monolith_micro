import time
from typing import Any

import pytest
from aiohttp import ClientConnectionError, ClientResponseError

from app.common.exceptions import ConnectionException
from app.core.aiohttp_exception_handler import external_request_exception_handler


@pytest.mark.asyncio
async def test_returns_result_when_no_error() -> None:
    @external_request_exception_handler()
    async def ok() -> str:
        return "fine"

    assert await ok() == "fine"


@pytest.mark.asyncio
async def test_raise_exception_type_reraises_when_is_raise_true() -> None:
    @external_request_exception_handler(raise_exception=ConnectionException)
    async def boom() -> None:
        raise ConnectionException("already ours")

    with pytest.raises(ConnectionException):
        await boom()


@pytest.mark.asyncio
async def test_raise_exception_type_returns_fallback_when_is_raise_false() -> None:
    @external_request_exception_handler(fallback="fallback-value", is_raise=False)
    async def boom() -> None:
        raise ConnectionException("already ours")

    assert await boom() == "fallback-value"


@pytest.mark.asyncio
async def test_client_connection_error_wrapped_when_is_raise_true() -> None:
    @external_request_exception_handler()
    async def boom() -> None:
        raise ClientConnectionError("network down")

    with pytest.raises(ConnectionException):
        await boom()


@pytest.mark.asyncio
async def test_client_connection_error_returns_fallback_when_is_raise_false() -> None:
    @external_request_exception_handler(fallback=None, is_raise=False)
    async def boom() -> Any:
        raise ClientConnectionError("network down")

    assert await boom() is None


@pytest.mark.asyncio
async def test_client_response_error_wrapped(mocker) -> None:
    @external_request_exception_handler()
    async def boom() -> None:
        raise ClientResponseError(request_info=mocker.Mock(), history=())

    with pytest.raises(ConnectionException):
        await boom()


@pytest.mark.asyncio
async def test_unexpected_exception_wrapped_when_is_raise_true() -> None:
    @external_request_exception_handler()
    async def boom() -> None:
        raise ValueError("totally unrelated")

    with pytest.raises(ConnectionException):
        await boom()


@pytest.mark.asyncio
async def test_unexpected_exception_returns_fallback_when_is_raise_false() -> None:
    @external_request_exception_handler(fallback="default", is_raise=False)
    async def boom() -> str:
        raise ValueError("totally unrelated")

    assert await boom() == "default"


@pytest.mark.asyncio
async def test_slow_call_logs_warning(mocker) -> None:
    times = iter([0.0, 2.0])
    mocker.patch.object(time, "time", side_effect=lambda: next(times))
    warning = mocker.patch("app.core.aiohttp_exception_handler.logger.warning")

    @external_request_exception_handler()
    async def slow() -> str:
        return "done"

    assert await slow() == "done"
    assert any("Slow request" in call.args[0] for call in warning.call_args_list)
