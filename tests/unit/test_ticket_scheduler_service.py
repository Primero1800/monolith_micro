from unittest.mock import AsyncMock

import pytest

from app.common.enums import TicketStatusEnum
from app.models.ticket import Ticket
from app.services.base import BaseDeps
from app.services.ticket_scheduler_service import TicketSchedulerService
from app.services.ticket_service import TicketService


def _make_ticket(ticket_id: int) -> Ticket:
    """Build a minimal claimed Ticket with the given id, for scheduler tests"""
    ticket = Ticket(
        raw_text="текст", normalized_text="текст", status=TicketStatusEnum.PROCESSING
    )
    ticket.id = ticket_id
    return ticket


@pytest.mark.asyncio
async def test_run_claims_and_processes_a_batch(mocker) -> None:
    """run() claims a batch using the configured settings and processes each ticket"""
    tickets = [_make_ticket(1), _make_ticket(2)]
    claim_pending = mocker.patch.object(
        TicketService, "claim_pending", new_callable=AsyncMock, return_value=tickets
    )
    process_pending = mocker.patch.object(
        TicketService, "process_pending", new_callable=AsyncMock
    )
    scheduler = TicketSchedulerService(
        base_deps=BaseDeps(uow_factory=None, ai_client=None)
    )

    await scheduler.run()

    claim_pending.assert_awaited_once()
    assert process_pending.await_count == 2
    processed_ticket_ids = {call.args[0].id for call in process_pending.await_args_list}
    assert processed_ticket_ids == {1, 2}


@pytest.mark.asyncio
async def test_run_does_nothing_when_no_tickets_are_pending(mocker) -> None:
    """run() is a no-op (no process_pending calls) when claim_pending finds nothing"""
    mocker.patch.object(
        TicketService, "claim_pending", new_callable=AsyncMock, return_value=[]
    )
    process_pending = mocker.patch.object(
        TicketService, "process_pending", new_callable=AsyncMock
    )
    scheduler = TicketSchedulerService(
        base_deps=BaseDeps(uow_factory=None, ai_client=None)
    )

    await scheduler.run()

    process_pending.assert_not_called()


@pytest.mark.asyncio
async def test_run_logs_one_failure_without_stopping_the_others(mocker) -> None:
    """One ticket raising during process_pending doesn't stop the rest of the batch"""
    tickets = [_make_ticket(1), _make_ticket(2), _make_ticket(3)]
    mocker.patch.object(
        TicketService, "claim_pending", new_callable=AsyncMock, return_value=tickets
    )

    async def _process(ticket: Ticket) -> Ticket:
        """Fail only for ticket id 2, succeed for the others"""
        if ticket.id == 2:
            raise RuntimeError("boom")
        return ticket

    process_pending = mocker.patch.object(
        TicketService, "process_pending", side_effect=_process
    )
    error_log = mocker.patch("app.services.ticket_scheduler_service.logger.error")
    scheduler = TicketSchedulerService(
        base_deps=BaseDeps(uow_factory=None, ai_client=None)
    )

    await scheduler.run()

    assert process_pending.await_count == 3
    error_log.assert_called_once()
    assert "2" in error_log.call_args.args[0]
