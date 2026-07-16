from unittest.mock import AsyncMock

import pytest

from app.lifecycle import AppLifecycle


@pytest.mark.asyncio
async def test_on_startup_initializes_db_and_aiohttp_session(mocker) -> None:
    init_db = mocker.patch("app.lifecycle.initialize_db", new_callable=AsyncMock)
    fake_session = AsyncMock()
    get_session = mocker.patch(
        "app.lifecycle.get_aiohttp_session", new_callable=AsyncMock
    )
    get_session.return_value = fake_session

    lifecycle = AppLifecycle(app=None)  # type: ignore[arg-type]
    await lifecycle.on_startup()

    init_db.assert_awaited_once()
    get_session.assert_awaited_once()
    assert lifecycle.aiohttp_session is fake_session


@pytest.mark.asyncio
async def test_on_startup_raises_runtime_error_when_db_init_fails(mocker) -> None:
    mocker.patch(
        "app.lifecycle.initialize_db",
        new_callable=AsyncMock,
        side_effect=ConnectionError("db unreachable"),
    )

    lifecycle = AppLifecycle(app=None)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="Core application initialization failed"):
        await lifecycle.on_startup()


@pytest.mark.asyncio
async def test_on_shutdown_closes_aiohttp_session_if_present(mocker) -> None:
    mocker.patch("app.lifecycle.initialize_db", new_callable=AsyncMock)
    lifecycle = AppLifecycle(app=None)  # type: ignore[arg-type]
    lifecycle.aiohttp_session = AsyncMock()

    await lifecycle.on_shutdown()

    lifecycle.aiohttp_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_shutdown_is_safe_without_a_session() -> None:
    lifecycle = AppLifecycle(app=None)  # type: ignore[arg-type]

    await lifecycle.on_shutdown()
