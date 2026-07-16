import logging

from app.common.enums import TicketStatusEnum
from app.common.logging import log_decorator
from app.models.ticket import Ticket
from app.services.base import BaseService
from app.utils.text_normalization import normalize_text


class TicketService(BaseService):
    """Orchestrates ticket classification: dedup check, regex fast-path, LLM fallback"""

    @log_decorator(level=logging.INFO)
    async def analyze(self, text: str) -> None:
        """Classify and summarize a ticket text, chaining the classification steps"""
        # 1. Clean up the text so it's easy to compare and classify
        normalized_text = normalize_text(text)

        # 2. Save the ticket right away so we don't lose it if something fails later
        await self._save_processing_ticket(text, normalized_text)

        # 3. If the text is empty or meaningless, just mark it "other" and stop

        # 4. If we've already answered this exact text before, reuse that answer

        # 5. If the text obviously matches one category, skip the LLM and use a template

        # 6. Otherwise ask the LLM to classify and summarize, then save the result
        raise NotImplementedError

    async def _save_processing_ticket(self, raw_text: str, normalized_text: str) -> Ticket:
        """Persist the incoming ticket immediately, before any classification attempt"""
        async with self.uow_factory as uow:
            return await uow.ticket_repository.create(
                Ticket(
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    status=TicketStatusEnum.PROCESSING,
                )
            )
