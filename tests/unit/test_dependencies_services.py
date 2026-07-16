from unittest.mock import AsyncMock

import pytest

from app.dependencies import services as services_module


@pytest.mark.asyncio
async def test_get_base_deps_standalone_assembles_deps_outside_fastapi(mocker) -> None:
    """get_base_deps_standalone() builds BaseDeps by awaiting each infra dependency directly"""
    fake_uow_factory = object()
    fake_session = object()
    fake_ai_client = object()
    mocker.patch.object(
        services_module,
        "get_uow_factory",
        new_callable=AsyncMock,
        return_value=fake_uow_factory,
    )
    mocker.patch.object(
        services_module,
        "get_aiohttp_session",
        new_callable=AsyncMock,
        return_value=fake_session,
    )
    get_ai_client = mocker.patch.object(
        services_module,
        "get_ai_client",
        new_callable=AsyncMock,
        return_value=fake_ai_client,
    )

    base_deps = await services_module.get_base_deps_standalone()

    assert base_deps.uow_factory is fake_uow_factory
    assert base_deps.ai_client is fake_ai_client
    get_ai_client.assert_awaited_once_with(fake_session)
