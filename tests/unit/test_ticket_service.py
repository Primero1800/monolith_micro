import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.adapters.ai_client import ChatResult
from app.common.enums import TicketCategoryEnum, TicketPriorityEnum, TicketStatusEnum
from app.models.ticket import Ticket
from app.services.base import BaseDeps
from app.services.ticket_service import TicketService


class _FakeUow:
    """Minimal object exposing the ticket_repository attribute TicketService reads"""

    def __init__(self, ticket_repository: Any) -> None:
        """Store the (mocked) repository this fake unit of work exposes"""
        self.ticket_repository = ticket_repository


class _FakeUowFactory:
    """Fake UnitOfWork usable as an async context manager, mirroring app.uow.UnitOfWork's shape"""

    def __init__(self, ticket_repository: Any) -> None:
        """Wrap the given repository in a fake unit of work and reset the enter counter"""
        self._uow = _FakeUow(ticket_repository)
        self.enter_count = 0

    async def __aenter__(self) -> _FakeUow:
        """Track entry count and return the fake unit of work"""
        self.enter_count += 1
        return self._uow

    async def __aexit__(self, *exc: Any) -> bool:
        """Do not suppress any exception raised inside the `async with` block"""
        return False


def _make_ticket(**overrides: Any) -> Ticket:
    """Build a Ticket ORM instance with sensible defaults, overriding any given fields"""
    ticket = Ticket(
        raw_text=overrides.pop("raw_text", "хочу снять квартиру"),
        normalized_text=overrides.pop("normalized_text", "хочу снять квартиру"),
        status=overrides.pop("status", TicketStatusEnum.PROCESSING),
    )
    ticket.id = overrides.pop("id", 1)
    for key, value in overrides.items():
        setattr(ticket, key, value)
    return ticket


def _make_service(ticket_repository: Any, ai_client: Any = None) -> TicketService:
    """Build a TicketService wired to a fake UnitOfWork factory and the given AI client"""
    uow_factory = _FakeUowFactory(ticket_repository)
    base_deps = BaseDeps(uow_factory=uow_factory, ai_client=ai_client)
    return TicketService(base_deps=base_deps)


@pytest.mark.asyncio
async def test_analyze_degenerate_text_short_circuits_without_llm() -> None:
    """Empty/punctuation-only text is marked "other" immediately, without touching the LLM or dedup lookup"""
    repo = AsyncMock()
    repo.create.return_value = _make_ticket()
    repo.mark_ready.return_value = _make_ticket(
        status=TicketStatusEnum.READY, category=TicketCategoryEnum.OTHER
    )
    ai_client = AsyncMock()
    service = _make_service(repo, ai_client)

    result = await service.analyze("   !!! ...")

    assert result.status == TicketStatusEnum.READY
    assert result.category == TicketCategoryEnum.OTHER
    repo.mark_ready.assert_awaited_once()
    ai_client.chat.assert_not_called()
    repo.find_ready_by_normalized_text.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_duplicate_reuses_previous_result_without_llm() -> None:
    """A normalized-text match against a previous ready ticket is copied, skipping the LLM call"""
    repo = AsyncMock()
    repo.create.return_value = _make_ticket(id=2)
    repo.find_ready_by_normalized_text.return_value = _make_ticket(
        id=1,
        status=TicketStatusEnum.READY,
        category=TicketCategoryEnum.RENT,
        summary="Клиент хочет снять квартиру",
        priority=TicketPriorityEnum.MEDIUM,
        entities={"budget": "40000"},
    )
    repo.mark_ready.return_value = _make_ticket(
        id=2,
        status=TicketStatusEnum.READY,
        category=TicketCategoryEnum.RENT,
        summary="Клиент хочет снять квартиру",
    )
    ai_client = AsyncMock()
    service = _make_service(repo, ai_client)

    result = await service.analyze("Хочу снять квартиру")

    assert result.category == TicketCategoryEnum.RENT
    repo.find_ready_by_normalized_text.assert_awaited_once()
    mark_ready_call = repo.mark_ready.await_args
    assert mark_ready_call.args[0] == 2
    assert mark_ready_call.args[1].ai_used is False
    ai_client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_no_duplicate_falls_through_to_llm_success() -> None:
    """With no dedup match and no regex fast-path hit, the LLM is called and its result persisted"""
    repo = AsyncMock()
    repo.create.return_value = _make_ticket(id=3)
    repo.find_ready_by_normalized_text.return_value = None
    repo.mark_ready.return_value = _make_ticket(
        id=3, status=TicketStatusEnum.READY, category=TicketCategoryEnum.SALE
    )
    llm_payload = json.dumps(
        {
            "category": "sale",
            "summary": "Клиент интересуется компанией",
            "priority": "high",
            "entities": {"district": "центр"},
        }
    )
    ai_client = AsyncMock()
    ai_client.chat.return_value = ChatResult(
        content=llm_payload, prompt_tokens=100, completion_tokens=50, total_tokens=150
    )
    service = _make_service(repo, ai_client)

    # No category keyword and no dedup match, so this must fall through to the LLM
    result = await service.analyze(
        "Добрый день, подскажите пожалуйста по вашей компании"
    )

    assert result.category == TicketCategoryEnum.SALE
    ai_client.chat.assert_awaited_once()
    mark_ready_call = repo.mark_ready.await_args
    classification = mark_ready_call.args[1]
    assert classification.ai_used is True
    assert classification.prompt_tokens == 100
    assert classification.completion_tokens == 50
    assert classification.llm_response_time_ms is not None
    repo.mark_failed.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_short_unambiguous_text_uses_regex_fast_path_without_llm() -> (
    None
):
    """Short text matching exactly one category's keyword is classified without calling the LLM"""
    repo = AsyncMock()
    repo.create.return_value = _make_ticket(id=8)
    repo.find_ready_by_normalized_text.return_value = None
    repo.mark_ready.return_value = _make_ticket(
        id=8, status=TicketStatusEnum.READY, category=TicketCategoryEnum.RENT
    )
    ai_client = AsyncMock()
    service = _make_service(repo, ai_client)

    result = await service.analyze("хочу снять квартиру")

    assert result.category == TicketCategoryEnum.RENT
    ai_client.chat.assert_not_called()
    mark_ready_call = repo.mark_ready.await_args
    classification = mark_ready_call.args[1]
    assert classification.ai_used is False
    assert classification.summary == "хочу снять квартиру"


@pytest.mark.asyncio
async def test_analyze_ambiguous_short_text_falls_through_to_llm() -> None:
    """Short text matching two categories' keywords is not guessed and goes to the LLM instead"""
    repo = AsyncMock()
    repo.create.return_value = _make_ticket(id=9)
    repo.find_ready_by_normalized_text.return_value = None
    repo.mark_ready.return_value = _make_ticket(
        id=9, status=TicketStatusEnum.READY, category=TicketCategoryEnum.OTHER
    )
    ai_client = AsyncMock()
    ai_client.chat.return_value = ChatResult(
        content=json.dumps(
            {"category": "other", "summary": "неоднозначный запрос", "priority": "low"}
        ),
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )
    service = _make_service(repo, ai_client)

    await service.analyze("хочу снять и купить квартиру")

    ai_client.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_llm_no_content_marks_failed() -> None:
    """A ChatResult with no content (transport failure) marks the ticket failed, not ready"""
    repo = AsyncMock()
    repo.create.return_value = _make_ticket(id=4)
    repo.find_ready_by_normalized_text.return_value = None
    repo.mark_failed.return_value = _make_ticket(id=4, status=TicketStatusEnum.FAILED)
    ai_client = AsyncMock()
    ai_client.chat.return_value = ChatResult(
        content=None, prompt_tokens=0, completion_tokens=0, total_tokens=0
    )
    service = _make_service(repo, ai_client)

    result = await service.analyze("что-то про недвижимость")

    assert result.status == TicketStatusEnum.FAILED
    repo.mark_failed.assert_awaited_once()
    assert repo.mark_failed.await_args.args[0] == 4
    repo.mark_ready.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_llm_invalid_json_marks_failed() -> None:
    """Content that doesn't match LLMClassificationOutput's schema marks the ticket failed"""
    repo = AsyncMock()
    repo.create.return_value = _make_ticket(id=5)
    repo.find_ready_by_normalized_text.return_value = None
    repo.mark_failed.return_value = _make_ticket(id=5, status=TicketStatusEnum.FAILED)
    ai_client = AsyncMock()
    ai_client.chat.return_value = ChatResult(
        content='{"category": "not_a_real_category"}',
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )
    service = _make_service(repo, ai_client)

    result = await service.analyze("что-то непонятное")

    assert result.status == TicketStatusEnum.FAILED
    repo.mark_failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_llm_unexpected_error_marks_failed_as_safety_net() -> None:
    """An unrelated exception from the AI client is caught by the safety net and marks failed"""
    repo = AsyncMock()
    repo.create.return_value = _make_ticket(id=6)
    repo.find_ready_by_normalized_text.return_value = None
    repo.mark_failed.return_value = _make_ticket(id=6, status=TicketStatusEnum.FAILED)
    ai_client = AsyncMock()
    ai_client.chat.side_effect = RuntimeError("totally unexpected")
    service = _make_service(repo, ai_client)

    result = await service.analyze("что угодно")

    assert result.status == TicketStatusEnum.FAILED
    repo.mark_failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_persists_ticket_before_classification() -> None:
    """The ticket is saved as processing before any classification step runs, so it's never lost"""
    repo = AsyncMock()
    repo.create.return_value = _make_ticket(id=7)
    repo.find_ready_by_normalized_text.return_value = None
    repo.mark_failed.return_value = _make_ticket(id=7, status=TicketStatusEnum.FAILED)
    ai_client = AsyncMock()
    ai_client.chat.return_value = ChatResult(
        content=None, prompt_tokens=0, completion_tokens=0, total_tokens=0
    )
    service = _make_service(repo, ai_client)

    await service.analyze("любой текст")

    repo.create.assert_awaited_once()
    created_ticket = repo.create.await_args.args[0]
    assert created_ticket.status == TicketStatusEnum.PROCESSING
    assert created_ticket.raw_text == "любой текст"
