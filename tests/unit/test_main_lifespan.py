from unittest.mock import AsyncMock

import pytest

from app.main import app, lifespan


@pytest.mark.asyncio
async def test_lifespan_runs_startup_then_shutdown(mocker) -> None:
    startup = mocker.patch(
        "app.main.AppLifecycle.on_startup", new_callable=AsyncMock
    )
    shutdown = mocker.patch(
        "app.main.AppLifecycle.on_shutdown", new_callable=AsyncMock
    )

    async with lifespan(app):
        startup.assert_awaited_once()
        shutdown.assert_not_awaited()

    shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_runs_shutdown_even_if_body_raises(mocker) -> None:
    mocker.patch("app.main.AppLifecycle.on_startup", new_callable=AsyncMock)
    shutdown = mocker.patch(
        "app.main.AppLifecycle.on_shutdown", new_callable=AsyncMock
    )

    with pytest.raises(ValueError):
        async with lifespan(app):
            raise ValueError("body failed")

    shutdown.assert_awaited_once()
