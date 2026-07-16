import asyncio

import pytest

from app.common.exceptions import DBHealthCheckError
from app.services.base import BaseDeps
from app.services.health_check_service import HealthCheckService


class _FakeUow:
    """Minimal async context manager whose session.execute is a no-op"""

    async def __aenter__(self) -> "_FakeUow":
        """Return self as the entered unit of work"""
        return self

    async def __aexit__(self, *exc: object) -> bool:
        """Do not suppress any exception raised inside the `async with` block"""
        return False

    class _Session:
        """Stub session exposing an execute() coroutine"""

        async def execute(self, *args: object, **kwargs: object) -> None:
            """Pretend to run the query and return nothing"""
            return None

    session = _Session()


def _make_service(uow_factory: object) -> HealthCheckService:
    """Build a HealthCheckService wired to the given fake UnitOfWork factory"""
    return HealthCheckService(
        base_deps=BaseDeps(uow_factory=uow_factory, ai_client=None)
    )


@pytest.mark.asyncio
async def test_check_db_status_raises_db_health_check_error_on_timeout(mocker) -> None:
    """A probe that exceeds HEALTH_CHECK_TIMEOUT_SEC is reported as DBHealthCheckError"""

    async def _time_out(coro: object, timeout: float) -> None:
        """Stand-in for asyncio.wait_for: close the pending coroutine, then time out

        Closing it avoids a "coroutine was never awaited" RuntimeWarning — the
        real coro argument is created before wait_for ever runs, so a bare
        side_effect=TimeoutError would leak it unawaited.
        """
        coro.close()  # type: ignore[attr-defined]
        raise asyncio.TimeoutError

    mocker.patch(
        "app.services.health_check_service.asyncio.wait_for",
        side_effect=_time_out,
    )
    service = _make_service(_FakeUow())

    with pytest.raises(DBHealthCheckError):
        await service.check_db_status()
