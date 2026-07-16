import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_success(async_client: AsyncClient) -> None:
    """Healthcheck returns 200 when the DB is reachable"""
    response = await async_client.get("/health_check")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_check_db_fail(async_client: AsyncClient, mocker) -> None:
    """Healthcheck returns 503 when opening a DB session fails"""
    from app.uow import UnitOfWork

    mocker.patch.object(
        UnitOfWork, "__aenter__", side_effect=Exception("DB connection failed")
    )
    response = await async_client.get("/health_check")
    assert response.status_code == 503
