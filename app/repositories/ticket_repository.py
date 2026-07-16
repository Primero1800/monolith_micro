import logging
from collections.abc import Sequence
from datetime import timedelta

from sqlalchemy import and_, case, cast, func, or_, select, update

from app.common.enums import TicketStatusEnum
from app.common.logging import log_decorator
from app.core.config import settings
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
    async def get_by_id(self, ticket_id: int) -> Ticket | None:
        """Fetch a ticket by its id

        :param:
            ticket_id: id of the ticket to fetch

        :returns:
            ticket: the matching Ticket, or None if not found
        """
        stmt = select(Ticket).where(Ticket.id == ticket_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

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
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                llm_response_time_ms=result.llm_response_time_ms,
            )
            .returning(Ticket)
        )
        updated = await self._session.execute(stmt)
        return updated.scalar_one()

    @log_decorator(level=logging.DEBUG)
    async def mark_failed(self, ticket_id: int, error_message: str) -> Ticket:
        """Mark a ticket as failed and bump its retry counter, or dead-letter it past the cap

        :param:
            ticket_id: id of the ticket to update
            error_message: description of what went wrong, for admin visibility

        :returns:
            ticket: the updated Ticket, with fresh values read back from the database
        """
        new_retries = Ticket.retries + 1
        stmt = (
            update(Ticket)
            .where(Ticket.id == ticket_id)
            .values(
                status=cast(
                    case(
                        (
                            new_retries >= settings.MAX_RETRIES,
                            TicketStatusEnum.DEAD_LETTER.name,
                        ),
                        else_=TicketStatusEnum.FAILED.name,
                    ),
                    Ticket.status.type,
                ),
                retries=new_retries,
                error_message=error_message,
            )
            .returning(Ticket)
        )
        updated = await self._session.execute(stmt)
        return updated.scalar_one()

    @log_decorator(level=logging.DEBUG)
    async def claim_pending(
        self, batch_size: int, stuck_after_sec: int
    ) -> Sequence[Ticket]:
        """Atomically claim a batch of tickets for the scheduler to (re)classify

        Candidates are draft/failed, or stuck processing (crashed mid-pipeline).
        Selecting and marking `processing` happen in one statement (the inner
        SELECT locks rows with FOR UPDATE SKIP LOCKED, the outer UPDATE flips
        their status) — no ticket can be claimed twice, regardless of how many
        scheduler workers are running.

        :param:
            batch_size: max number of tickets to claim in one tick
            stuck_after_sec: how long a ticket may sit in `processing` before it's
                considered abandoned (e.g. the process crashed mid-pipeline)

        :returns:
            tickets: the claimed tickets, now marked `processing`
        """
        stuck_threshold = func.now() - timedelta(seconds=stuck_after_sec)
        candidate_ids = (
            select(Ticket.id)
            .where(
                or_(
                    Ticket.status.in_(
                        [TicketStatusEnum.DRAFT, TicketStatusEnum.FAILED]
                    ),
                    and_(
                        Ticket.status == TicketStatusEnum.PROCESSING,
                        Ticket.updated_at < stuck_threshold,
                    ),
                )
            )
            .order_by(Ticket.created_at.asc())
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        stmt = (
            update(Ticket)
            .where(Ticket.id.in_(candidate_ids))
            .values(status=TicketStatusEnum.PROCESSING)
            .returning(Ticket)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()
