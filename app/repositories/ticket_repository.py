import logging

from sqlalchemy import select

from app.common.enums import TicketStatusEnum
from app.common.logging import log_decorator
from app.models.ticket import Ticket
from app.repositories.base_repository import BaseRepository
from app.repositories.repository_error_handler import repository_error_handler


@repository_error_handler
class TicketRepository(BaseRepository):
    """Repository for creating and searching Ticket records in PostgreSQL"""

    @log_decorator(level=logging.DEBUG)
    async def create(self, ticket: Ticket) -> Ticket:
        """Persist a new ticket record

        :param:
            ticket: the Ticket ORM instance to insert

        :returns:
            ticket: the persisted Ticket, with server-generated defaults flushed
        """
        self._session.add(ticket)
        await self._session.flush()
        return ticket

    @log_decorator(level=logging.DEBUG)
    async def find_ready_by_normalized_text(
        self, normalized_text: str
    ) -> Ticket | None:
        """Find the most recent already-processed ticket with the same normalized text

        Used to skip a redundant LLM call when an identical request (ignoring
        case/punctuation) has already been classified successfully.

        :param:
            normalized_text: lowercase, punctuation-stripped ticket text

        :returns:
            ticket: the most recent matching `ready` ticket, or None if none found
        """
        stmt = (
            select(Ticket)
            .where(
                Ticket.normalized_text == normalized_text,
                Ticket.status == TicketStatusEnum.READY,
            )
            .order_by(Ticket.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
