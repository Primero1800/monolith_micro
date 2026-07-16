import socket

import pytest
import sqlalchemy.exc

from app.uow import UnitOfWork


class _FakeSession:
    def __init__(self, commit_error: Exception | None = None) -> None:
        self._commit_error = commit_error
        self.committed = False
        self.rolled_back = False
        self.closed = False

    async def commit(self) -> None:
        if self._commit_error is not None:
            raise self._commit_error
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_commits_on_clean_exit() -> None:
    session = _FakeSession()
    uow = UnitOfWork(session_factory=lambda: session)

    async with uow:
        pass

    assert session.committed is True
    assert session.closed is True


@pytest.mark.asyncio
async def test_attaches_ticket_repository_on_enter() -> None:
    session = _FakeSession()
    uow = UnitOfWork(session_factory=lambda: session)

    async with uow as entered:
        assert entered is uow
        assert entered.ticket_repository is not None
        assert entered.ticket_repository._session is session


@pytest.mark.asyncio
async def test_rolls_back_and_reraises_when_body_raises() -> None:
    session = _FakeSession()
    uow = UnitOfWork(session_factory=lambda: session)

    with pytest.raises(ValueError):
        async with uow:
            raise ValueError("body failed")

    assert session.rolled_back is True
    assert session.closed is True
    assert session.committed is False


@pytest.mark.asyncio
async def test_rolls_back_and_reraises_when_commit_fails() -> None:
    session = _FakeSession(commit_error=sqlalchemy.exc.IntegrityError("s", {}, Exception()))
    uow = UnitOfWork(session_factory=lambda: session)

    with pytest.raises(sqlalchemy.exc.IntegrityError):
        async with uow:
            pass

    assert session.rolled_back is True
    assert session.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_type, exc_instance",
    [
        (sqlalchemy.exc.OperationalError, sqlalchemy.exc.OperationalError("s", {}, Exception())),
        (ConnectionRefusedError, ConnectionRefusedError("refused")),
        (socket.gaierror, socket.gaierror("dns failure")),
        (sqlalchemy.exc.ProgrammingError, sqlalchemy.exc.ProgrammingError("s", {}, Exception())),
        (sqlalchemy.exc.IntegrityError, sqlalchemy.exc.IntegrityError("s", {}, Exception())),
    ],
)
async def test_exit_classifies_incoming_exception_types_without_masking(
    exc_type: type, exc_instance: Exception
) -> None:
    session = _FakeSession()
    uow = UnitOfWork(session_factory=lambda: session)

    with pytest.raises(exc_type):
        async with uow:
            raise exc_instance

    assert session.rolled_back is True
    assert session.closed is True
