from types import SimpleNamespace

import pytest
from asyncpg import UniqueViolationError
from sqlalchemy.exc import IntegrityError

from app.common.exceptions import IntegrityDataException
from app.repositories.repository_error_handler import repository_error_handler


def _make_integrity_error(orig: Exception | None) -> IntegrityError:
    return IntegrityError("stmt", {}, orig)  # type: ignore[arg-type]


@repository_error_handler
class _FakeRepository:
    def __init__(self, to_raise: Exception | None = None, return_value: str = "ok") -> None:
        self._to_raise = to_raise
        self._return_value = return_value

    async def do_something(self) -> str:
        if self._to_raise is not None:
            raise self._to_raise
        return self._return_value


@pytest.mark.asyncio
async def test_passthrough_on_success() -> None:
    repo = _FakeRepository()
    assert await repo.do_something() == "ok"


@pytest.mark.asyncio
async def test_known_integrity_error_wrapped() -> None:
    orig = SimpleNamespace(pgcode=UniqueViolationError.sqlstate)
    repo = _FakeRepository(to_raise=_make_integrity_error(orig))

    with pytest.raises(IntegrityDataException):
        await repo.do_something()


@pytest.mark.asyncio
async def test_known_integrity_error_by_instance_type_wrapped() -> None:
    orig = UniqueViolationError()
    repo = _FakeRepository(to_raise=_make_integrity_error(orig))

    with pytest.raises(IntegrityDataException):
        await repo.do_something()


@pytest.mark.asyncio
async def test_unknown_integrity_error_reraised_as_is() -> None:
    orig = SimpleNamespace(pgcode="99999")
    repo = _FakeRepository(to_raise=_make_integrity_error(orig))

    with pytest.raises(IntegrityError):
        await repo.do_something()


@pytest.mark.asyncio
async def test_non_integrity_exception_propagates_untouched() -> None:
    repo = _FakeRepository(to_raise=ValueError("unrelated"))

    with pytest.raises(ValueError):
        await repo.do_something()


def test_dunder_methods_are_not_wrapped() -> None:
    # __init__ should stay untouched by the class decorator
    repo = _FakeRepository(return_value="hello")
    assert repo._return_value == "hello"
