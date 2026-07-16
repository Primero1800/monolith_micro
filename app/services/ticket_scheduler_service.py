import asyncio

from app.common.logging import logger
from app.core.config import settings
from app.services.base import BaseDeps
from app.services.ticket_service import TicketService


class TicketSchedulerService:
    """Background scheduler: picks up draft/failed/stuck tickets and classifies them"""

    def __init__(self, base_deps: BaseDeps) -> None:
        """Store base_deps to build a fresh TicketService for this tick"""
        self._base_deps = base_deps

    async def run(self) -> None:
        """Fetch a batch of pending tickets and classify them concurrently

        An exception in one ticket must not stop the others from being
        processed, so gather collects exceptions instead of propagating them.
        """
        ticket_service = TicketService(self._base_deps)
        tickets = await ticket_service.claim_pending(
            batch_size=settings.SCHEDULER_BATCH_SIZE,
            stuck_after_sec=settings.PROCESSING_TIMEOUT_SEC,
        )
        results = await asyncio.gather(
            *(ticket_service.process_pending(t) for t in tickets),
            return_exceptions=True,
        )
        for ticket, result in zip(tickets, results):
            if isinstance(result, Exception):
                logger.error(
                    f"[ticket_scheduler] ticket {ticket.id} failed: {result!r}"
                )
