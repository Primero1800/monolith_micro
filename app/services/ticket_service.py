import logging
import time

from pydantic import ValidationError

from app.adapters.ai_client import Message
from app.common.enums import TicketCategoryEnum, TicketPriorityEnum, TicketStatusEnum
from app.common.exceptions import ConnectionException
from app.common.logging import log_decorator, logger
from app.models.ticket import ClassificationResult, Ticket
from app.pyd.llm_classification import LLMClassificationOutput
from app.services.base import BaseService
from app.services.prompt_service import PromptService
from app.utils.text_normalization import is_degenerate_text, normalize_text


class TicketService(BaseService):
    """Orchestrates ticket classification: dedup check, regex fast-path, LLM fallback"""

    @log_decorator(level=logging.INFO)
    async def analyze(self, text: str) -> Ticket:
        """Classify and summarize a ticket text, chaining the classification steps"""
        # 1. Clean up the text so it's easy to compare and classify
        normalized_text = normalize_text(text)

        # 2. Save the ticket right away so we don't lose it if something fails later
        ticket = await self._save_processing_ticket(text, normalized_text)

        # 3. If the text is empty or meaningless, just mark it "other" and stop
        degenerate = await self._check_degenerate(ticket.id, normalized_text)
        if degenerate is not None:
            return degenerate

        # 4. If we've already answered this exact text before, reuse that answer
        duplicate = await self._check_duplicate(ticket.id, normalized_text)
        if duplicate is not None:
            return duplicate

        # 5. Skipped: only helps very short, single-topic messages, not worth the complexity

        # 6. Otherwise ask the LLM to classify and summarize, then save the result
        return await self._classify_with_llm(ticket.id, text)

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

    async def _check_degenerate(self, ticket_id: int, normalized_text: str) -> Ticket | None:
        """Close out a ticket immediately if there's nothing meaningful to classify"""
        if not is_degenerate_text(normalized_text):
            return None

        result = ClassificationResult(
            category=TicketCategoryEnum.OTHER,
            summary="Пустое или нераспознаваемое обращение",
            priority=TicketPriorityEnum.LOW,
            entities=None,
            ai_used=False,
        )
        async with self.uow_factory as uow:
            return await uow.ticket_repository.mark_ready(ticket_id, result)

    async def _check_duplicate(self, ticket_id: int, normalized_text: str) -> Ticket | None:
        """Reuse a previous ready ticket's result if its normalized text matches exactly"""
        async with self.uow_factory as uow:
            match = await uow.ticket_repository.find_ready_by_normalized_text(
                normalized_text
            )
            if match is None:
                return None

            result = ClassificationResult(
                category=match.category,
                summary=match.summary,
                priority=match.priority,
                entities=match.entities,
                ai_used=False,
            )
            return await uow.ticket_repository.mark_ready(ticket_id, result)

    async def _classify_with_llm(self, ticket_id: int, text: str) -> Ticket:
        """Ask the LLM to classify the ticket, then persist the result or the failure

        The LLM call/parsing (no DB access) and the DB write are kept separate,
        so a genuine DB problem in mark_ready/mark_failed is never mislabeled
        as an LLM failure.
        """
        result = await self._request_llm_classification(text)

        if result is None:
            return await self._mark_failed(ticket_id, "LLM classification failed")

        async with self.uow_factory as uow:
            return await uow.ticket_repository.mark_ready(ticket_id, result)

    async def _request_llm_classification(self, text: str) -> ClassificationResult | None:
        """Call the LLM and parse its response into a classification result

        Returns None on ConnectionException (no response) or ValidationError
        (response doesn't match our schema) — both expected failure modes,
        handled identically by the caller: mark failed, no retry here, the
        scheduler's own retry mechanism picks it up later.
        """
        messages: list[Message] = [
            {"role": "system", "content": PromptService.get_ticket_classification_prompt()},
            {"role": "user", "content": text},
        ]

        started_at = time.monotonic()
        try:
            chat_result = await self.ai_client.chat(messages, json_mode=True)
            if chat_result.content is None:
                raise ConnectionException("LLM call returned no content")

            parsed = LLMClassificationOutput.model_validate_json(chat_result.content)
        except (ConnectionException, ValidationError) as exc:
            logger.error(f"LLM classification request failed: {exc!r}")
            return None
        except Exception as exc:
            logger.error(f"Unexpected error during LLM classification: {exc!r}")
            return None

        return ClassificationResult(
            category=parsed.category,
            summary=parsed.summary,
            priority=parsed.priority,
            entities=parsed.entities,
            ai_used=True,
            prompt_tokens=chat_result.prompt_tokens,
            completion_tokens=chat_result.completion_tokens,
            llm_response_time_ms=int((time.monotonic() - started_at) * 1000),
        )

    async def _mark_failed(self, ticket_id: int, error_message: str) -> Ticket:
        """Record a failed classification attempt so it can be retried or reviewed later"""
        async with self.uow_factory as uow:
            return await uow.ticket_repository.mark_failed(ticket_id, error_message)
