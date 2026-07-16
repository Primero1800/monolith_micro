from unittest.mock import AsyncMock

import pytest

from app.services.ticket_scheduler_service import TicketSchedulerService
from app.utils.ticket_scheduler_func import run_ticket_scheduler


@pytest.mark.asyncio
async def test_run_ticket_scheduler_builds_deps_and_runs_one_tick(mocker) -> None:
    """run_ticket_scheduler() assembles standalone deps and delegates to TicketSchedulerService.run()"""
    fake_base_deps = object()
    get_base_deps_standalone = mocker.patch(
        "app.utils.ticket_scheduler_func.get_base_deps_standalone",
        new_callable=AsyncMock,
        return_value=fake_base_deps,
    )
    run = mocker.patch.object(TicketSchedulerService, "run", new_callable=AsyncMock)

    await run_ticket_scheduler()

    get_base_deps_standalone.assert_awaited_once()
    run.assert_awaited_once()
