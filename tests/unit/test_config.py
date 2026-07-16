import logging

import pytest

from app.core.config import Settings


def _make_settings(**overrides: object) -> Settings:
    base = dict(
        APP_NAME="test_monolith",
        APP_VERSION="0.1.0",
        POSTGRES_HOST="db",
        POSTGRES_USER="postgres",
        POSTGRES_PASSWORD="postgres",
        POSTGRES_PORT=5432,
        POSTGRES_DB="test_monolith_db",
        POOL_SIZE=10,
        MAX_OVERFLOW=10,
        LOG_LEVEL="DEBUG",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg,arg-type]


def test_database_url_format() -> None:
    settings = _make_settings()
    assert settings.database_url == (
        "postgresql+asyncpg://postgres:postgres@db:5432/test_monolith_db"
    )


@pytest.mark.parametrize(
    "level_name, expected",
    [
        ("DEBUG", logging.DEBUG),
        ("info", logging.INFO),
        ("Warning", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("critical", logging.CRITICAL),
    ],
)
def test_log_level_valid(level_name: str, expected: int) -> None:
    settings = _make_settings(LOG_LEVEL=level_name)
    assert settings.log_level == expected


def test_log_level_invalid_raises() -> None:
    settings = _make_settings(LOG_LEVEL="NOT_A_LEVEL")
    with pytest.raises(ValueError, match="Invalid log level"):
        _ = settings.log_level
