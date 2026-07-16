from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import get_session, get_session_maker, initialize_db, shutdown_db


@pytest.mark.asyncio
async def test_get_session_maker_returns_the_global_session_maker() -> None:
    """get_session_maker exposes the module-level async_sessionmaker singleton"""
    from app.core.database import async_database_session_maker

    assert await get_session_maker() is async_database_session_maker


@pytest.mark.asyncio
async def test_get_session_yields_and_closes_a_session(mocker) -> None:
    """get_session yields one session from the session maker and always closes it"""
    fake_session = AsyncMock()
    fake_session_cm = MagicMock()
    fake_session_cm.__aenter__.return_value = fake_session
    fake_session_cm.__aexit__.return_value = False
    mocker.patch(
        "app.core.database.async_database_session_maker",
        return_value=fake_session_cm,
    )

    async for session in get_session():
        assert session is fake_session

    fake_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_session_rolls_back_and_reraises_on_sqlalchemy_error(mocker) -> None:
    """get_session rolls back and re-raises when a SQLAlchemyError is thrown into it

    FastAPI drives generator dependencies with asend/athrow (via anyio), not a plain
    `async for` — a raise inside a plain for-loop body just aborts iteration and calls
    aclose(), never reaching the generator's own except block. athrow() reproduces
    what FastAPI actually does when the endpoint raises while a dependency is open.
    """
    fake_session = AsyncMock()
    fake_session_cm = MagicMock()
    fake_session_cm.__aenter__.return_value = fake_session
    fake_session_cm.__aexit__.return_value = False
    mocker.patch(
        "app.core.database.async_database_session_maker",
        return_value=fake_session_cm,
    )

    gen = get_session()
    session = await gen.asend(None)
    assert session is fake_session

    with pytest.raises(SQLAlchemyError):
        await gen.athrow(SQLAlchemyError("boom"))

    fake_session.rollback.assert_awaited_once()
    fake_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_db_runs_a_connectivity_check(mocker) -> None:
    """initialize_db opens a connection from the engine and executes a SELECT 1 probe"""
    fake_conn = AsyncMock()
    fake_conn_cm = MagicMock()
    fake_conn_cm.__aenter__.return_value = fake_conn
    fake_conn_cm.__aexit__.return_value = False
    fake_engine = MagicMock()
    fake_engine.begin.return_value = fake_conn_cm
    mocker.patch("app.core.database.engine", fake_engine)

    await initialize_db()

    fake_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_db_disposes_the_engine(mocker) -> None:
    """shutdown_db disposes of the module-level engine"""
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()
    mocker.patch("app.core.database.engine", fake_engine)

    await shutdown_db()

    fake_engine.dispose.assert_awaited_once()
