from types import SimpleNamespace

import pytest
from asyncpg import UniqueViolationError
from sqlalchemy.exc import IntegrityError

from app.common.exceptions import IntegrityDataException
from app.repositories.repository_error_handler import repository_error_handler


def _make_integrity_error(orig: Exception | None) -> IntegrityError:
    """Build a SQLAlchemy IntegrityError wrapping the given original driver exception"""
    return IntegrityError("stmt", {}, orig)  # type: ignore[arg-type]


@repository_error_handler
class _FakeRepository:
    """Minimal repository used to exercise the repository_error_handler class decorator"""

    def __init__(
        self, to_raise: Exception | None = None, return_value: str = "ok"
    ) -> None:
        """Configure what do_something() should raise or return"""
        self._to_raise = to_raise
        self._return_value = return_value

    async def do_something(self) -> str:
        """Either raise the configured exception or return the configured value"""
        if self._to_raise is not None:
            raise self._to_raise
        return self._return_value


@pytest.mark.asyncio
async def test_passthrough_on_success() -> None:
    """A successful repository call passes its return value through untouched"""
    repo = _FakeRepository()
    assert await repo.do_something() == "ok"


@pytest.mark.asyncio
async def test_known_integrity_error_wrapped() -> None:
    """A known Postgres constraint sqlstate is wrapped into IntegrityDataException"""
    orig = SimpleNamespace(pgcode=UniqueViolationError.sqlstate)
    repo = _FakeRepository(to_raise=_make_integrity_error(orig))

    with pytest.raises(IntegrityDataException):
        await repo.do_something()


@pytest.mark.asyncio
async def test_known_integrity_error_by_instance_type_wrapped() -> None:
    """A known violation recognized by instance type (not just sqlstate) is also wrapped"""
    orig = UniqueViolationError()
    repo = _FakeRepository(to_raise=_make_integrity_error(orig))

    with pytest.raises(IntegrityDataException):
        await repo.do_something()


@pytest.mark.asyncio
async def test_unknown_integrity_error_reraised_as_is() -> None:
    """An IntegrityError with an unrecognized sqlstate is re-raised untouched, not misclassified"""
    orig = SimpleNamespace(pgcode="99999")
    repo = _FakeRepository(to_raise=_make_integrity_error(orig))

    with pytest.raises(IntegrityError):
        await repo.do_something()


@pytest.mark.asyncio
async def test_non_integrity_exception_propagates_untouched() -> None:
    """Exceptions unrelated to IntegrityError are left completely alone"""
    repo = _FakeRepository(to_raise=ValueError("unrelated"))

    with pytest.raises(ValueError):
        await repo.do_something()


def test_dunder_methods_are_not_wrapped() -> None:
    """The class decorator skips dunder methods like __init__, leaving them unwrapped"""
    repo = _FakeRepository(return_value="hello")
    assert repo._return_value == "hello"
