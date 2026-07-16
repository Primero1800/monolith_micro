import time
from types import SimpleNamespace

import pytest
from fastapi import Response

from app.middleware import MonitoringMiddleware


def _fake_request() -> SimpleNamespace:
    """Minimal object exposing the method/url attributes dispatch() logs"""
    return SimpleNamespace(method="GET", url="http://test/tickets")


@pytest.mark.asyncio
async def test_dispatch_passes_through_response_on_success() -> None:
    """A successful call_next response is returned unchanged"""
    middleware = MonitoringMiddleware(app=None)  # type: ignore[arg-type]
    expected_response = Response(content="ok", status_code=200)

    async def call_next(request: object) -> Response:
        """Simulate a downstream handler that succeeds"""
        return expected_response

    response = await middleware.dispatch(_fake_request(), call_next)  # type: ignore[arg-type]

    assert response is expected_response


@pytest.mark.asyncio
async def test_dispatch_converts_unhandled_exception_to_500(mocker) -> None:
    """An unhandled exception from call_next is logged and converted to a JSON 500 response"""
    middleware = MonitoringMiddleware(app=None)  # type: ignore[arg-type]
    error_log = mocker.patch("app.middleware.logger.error")

    async def call_next(request: object) -> Response:
        """Simulate a downstream handler that raises"""
        raise RuntimeError("boom")

    response = await middleware.dispatch(_fake_request(), call_next)  # type: ignore[arg-type]

    assert response.status_code == 500
    assert b"Internal Server Error" in response.body
    error_log.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_logs_warning_on_slow_request(mocker) -> None:
    """A request whose elapsed time exceeds 1 second logs a SLOW warning but still returns normally"""
    middleware = MonitoringMiddleware(app=None)  # type: ignore[arg-type]
    times = iter([0.0, 2.0])
    mocker.patch.object(time, "time", side_effect=lambda: next(times))
    warning_log = mocker.patch("app.middleware.logger.warning")
    expected_response = Response(content="ok", status_code=200)

    async def call_next(request: object) -> Response:
        """Simulate a downstream handler whose elapsed time is faked to be slow"""
        return expected_response

    response = await middleware.dispatch(_fake_request(), call_next)  # type: ignore[arg-type]

    assert response is expected_response
    assert any("SLOW" in call.args[0] for call in warning_log.call_args_list)
