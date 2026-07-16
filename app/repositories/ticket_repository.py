import logging

from sqlalchemy import select, update

from app.common.enums import TicketStatusEnum
from app.common.logging import log_decorator
from app.models.ticket import ClassificationResult, Ticket
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

    @log_decorator(level=logging.DEBUG)
    async def mark_ready(self, ticket_id: int, result: ClassificationResult) -> Ticket:
        """Apply a classification result to a ticket and mark it ready

        :param:
            ticket_id: id of the ticket to update
            result: classification data to apply, from whichever pipeline step produced it

        :returns:
            ticket: the updated Ticket, with fresh values read back from the database
        """
        stmt = (
            update(Ticket)
            .where(Ticket.id == ticket_id)
            .values(
                status=TicketStatusEnum.READY,
                category=result.category,
                summary=result.summary,
                priority=result.priority,
                entities=result.entities,
                ai_used=result.ai_used,
            )
            .returning(Ticket)
        )
        updated = await self._session.execute(stmt)
        return updated.scalar_one()

    @log_decorator(level=logging.DEBUG)
    async def mark_failed(self, ticket_id: int, error_message: str) -> Ticket:
        """Mark a ticket as failed and bump its retry counter

        :param:
            ticket_id: id of the ticket to update
            error_message: description of what went wrong, for admin visibility

        :returns:
            ticket: the updated Ticket, with fresh values read back from the database
        """
        stmt = (
            update(Ticket)
            .where(Ticket.id == ticket_id)
            .values(
                status=TicketStatusEnum.FAILED,
                retries=Ticket.retries + 1,
                error_message=error_message,
            )
            .returning(Ticket)
        )
        updated = await self._session.execute(stmt)
        return updated.scalar_one()
