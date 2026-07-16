from app.common.logging import logger
from app.services.base import BaseService


class TicketSchedulerService(BaseService):
    """Background scheduler: picks up draft/failed/stuck tickets and classifies them

    Skeleton only for now — runs on schedule but doesn't do anything yet.
    """

    async def run(self) -> None:
        """Run one scheduler tick"""
        logger.info("[ticket_scheduler] tick")
