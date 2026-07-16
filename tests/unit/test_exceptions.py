from fastapi import status

from app.common.exceptions import (
    BaseCustomException,
    ConnectionException,
    DBHealthCheckError,
    IntegrityDataException,
)


def test_base_custom_exception_defaults() -> None:
    """Default status code is 503 and headers default to None"""
    exc = BaseCustomException(detail="oops")
    assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc.detail == "oops"
    assert exc.headers is None


def test_base_custom_exception_custom_status_and_headers() -> None:
    """status_code and headers are stored as passed, overriding the defaults"""
    exc = BaseCustomException(
        detail="oops", status_code=418, headers={"Retry-After": "5"}
    )
    assert exc.status_code == 418
    assert exc.headers == {"Retry-After": "5"}


def test_integrity_data_exception_strips_detail_prefix() -> None:
    """The raw Postgres DETAIL: prefix is stripped, leaving just the human-readable message"""
    raw = (
        "duplicate key value violates unique constraint "
        "DETAIL:  Key (id)=(1) already exists."
    )
    exc = IntegrityDataException(detail=raw)
    assert exc.detail == "Key (id)=(1) already exists."


def test_integrity_data_exception_keeps_detail_when_no_prefix() -> None:
    """Without a DETAIL: prefix, the original message passes through unchanged"""
    exc = IntegrityDataException(detail="something went wrong")
    assert exc.detail == "something went wrong"


def test_db_health_check_error_is_base_custom_exception() -> None:
    """DBHealthCheckError inherits BaseCustomException's status/detail/headers behavior"""
    exc = DBHealthCheckError(detail="db down")
    assert isinstance(exc, BaseCustomException)


def test_connection_exception_is_base_custom_exception() -> None:
    """ConnectionException inherits BaseCustomException's status/detail/headers behavior"""
    exc = ConnectionException(detail="network down")
    assert isinstance(exc, BaseCustomException)
