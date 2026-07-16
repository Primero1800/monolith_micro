import time
from typing import Any

import pytest
from aiohttp import ClientConnectionError, ClientResponseError

from app.common.exceptions import ConnectionException
from app.core.aiohttp_exception_handler import external_request_exception_handler


@pytest.mark.asyncio
async def test_returns_result_when_no_error() -> None:
    """A successful call passes its return value through untouched"""

    @external_request_exception_handler()
    async def ok() -> str:
        """Trivial wrapped function that always succeeds"""
        return "fine"

    assert await ok() == "fine"


@pytest.mark.asyncio
async def test_raise_exception_type_reraises_when_is_raise_true() -> None:
    """An exception of the configured raise_exception type is re-raised as-is, not wrapped"""

    @external_request_exception_handler(raise_exception=ConnectionException)
    async def boom() -> None:
        """Wrapped function that raises the configured exception type directly"""
        raise ConnectionException("already ours")

    with pytest.raises(ConnectionException):
        await boom()


@pytest.mark.asyncio
async def test_raise_exception_type_returns_fallback_when_is_raise_false() -> None:
    """With is_raise=False, the configured exception type is swallowed and fallback returned"""

    @external_request_exception_handler(fallback="fallback-value", is_raise=False)
    async def boom() -> None:
        """Wrapped function that raises the configured exception type directly"""
        raise ConnectionException("already ours")

    assert await boom() == "fallback-value"


@pytest.mark.asyncio
async def test_client_connection_error_wrapped_when_is_raise_true() -> None:
    """A ClientConnectionError is wrapped into raise_exception (default ConnectionException)"""

    @external_request_exception_handler()
    async def boom() -> None:
        """Wrapped function that raises a transport-level connection error"""
        raise ClientConnectionError("network down")

    with pytest.raises(ConnectionException):
        await boom()


@pytest.mark.asyncio
async def test_client_connection_error_returns_fallback_when_is_raise_false() -> None:
    """With is_raise=False, a ClientConnectionError is swallowed and fallback returned"""

    @external_request_exception_handler(fallback=None, is_raise=False)
    async def boom() -> Any:
        """Wrapped function that raises a transport-level connection error"""
        raise ClientConnectionError("network down")

    assert await boom() is None


@pytest.mark.asyncio
async def test_client_response_error_wrapped(mocker) -> None:
    """A ClientResponseError (bad HTTP status) is wrapped into raise_exception"""

    @external_request_exception_handler()
    async def boom() -> None:
        """Wrapped function that raises an HTTP-status-level error"""
        raise ClientResponseError(request_info=mocker.Mock(), history=())

    with pytest.raises(ConnectionException):
        await boom()


@pytest.mark.asyncio
async def test_unexpected_exception_wrapped_when_is_raise_true() -> None:
    """Any other unexpected exception is also wrapped into raise_exception, not left bare"""

    @external_request_exception_handler()
    async def boom() -> None:
        """Wrapped function that raises an exception unrelated to networking"""
        raise ValueError("totally unrelated")

    with pytest.raises(ConnectionException):
        await boom()


@pytest.mark.asyncio
async def test_unexpected_exception_returns_fallback_when_is_raise_false() -> None:
    """With is_raise=False, an unrelated exception is swallowed and fallback returned"""

    @external_request_exception_handler(fallback="default", is_raise=False)
    async def boom() -> str:
        """Wrapped function that raises an exception unrelated to networking"""
        raise ValueError("totally unrelated")

    assert await boom() == "default"


@pytest.mark.asyncio
async def test_slow_call_logs_warning(mocker) -> None:
    """A call exceeding 1 second logs a "Slow request" warning even though it still succeeds"""
    times = iter([0.0, 2.0])
    mocker.patch.object(time, "time", side_effect=lambda: next(times))
    warning = mocker.patch("app.core.aiohttp_exception_handler.logger.warning")

    @external_request_exception_handler()
    async def slow() -> str:
        """Wrapped function whose elapsed time is faked to exceed the slow-request threshold"""
        return "done"

    assert await slow() == "done"
    assert any("Slow request" in call.args[0] for call in warning.call_args_list)
