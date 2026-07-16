import pytest

from app.common.enums import TicketCategoryEnum, TicketPriorityEnum, TicketStatusEnum
from app.models.ticket import ClassificationResult, Ticket
from app.repositories.ticket_repository import TicketRepository


@pytest.mark.asyncio
async def test_create_persists_ticket_with_defaults(
    empty_db, test_session_maker
) -> None:
    """create() flushes a new ticket and returns it with server-side defaults populated"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        ticket = Ticket(
            raw_text="хочу снять квартиру",
            normalized_text="хочу снять квартиру",
            status=TicketStatusEnum.PROCESSING,
        )
        created = await repo.create(ticket)
        await session.commit()

        assert created.id is not None
        assert created.status == TicketStatusEnum.PROCESSING
        assert created.retries == 0
        assert created.ai_used is False
        assert created.prompt_tokens == 0
        assert created.completion_tokens == 0


@pytest.mark.asyncio
async def test_find_ready_by_normalized_text_returns_most_recent_match(
    empty_db, test_session_maker
) -> None:
    """Among several ready tickets sharing normalized_text, the most recently created one wins"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        await repo.create(
            Ticket(
                raw_text="a",
                normalized_text="дубликат текста",
                status=TicketStatusEnum.READY,
                category=TicketCategoryEnum.RENT,
                summary="старый",
                priority=TicketPriorityEnum.LOW,
            )
        )
        await session.commit()

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        await repo.create(
            Ticket(
                raw_text="b",
                normalized_text="дубликат текста",
                status=TicketStatusEnum.READY,
                category=TicketCategoryEnum.SALE,
                summary="новый",
                priority=TicketPriorityEnum.HIGH,
            )
        )
        await session.commit()

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        match = await repo.find_ready_by_normalized_text("дубликат текста")

        assert match is not None
        assert match.category == TicketCategoryEnum.SALE
        assert match.summary == "новый"


@pytest.mark.asyncio
async def test_find_ready_by_normalized_text_ignores_non_ready_status(
    empty_db, test_session_maker
) -> None:
    """A ticket still in `processing` is not returned as a dedup match, even with the same text"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        await repo.create(
            Ticket(
                raw_text="a",
                normalized_text="ещё обрабатывается",
                status=TicketStatusEnum.PROCESSING,
            )
        )
        await session.commit()

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        match = await repo.find_ready_by_normalized_text("ещё обрабатывается")

        assert match is None


@pytest.mark.asyncio
async def test_find_ready_by_normalized_text_no_match_at_all(
    empty_db, test_session_maker
) -> None:
    """Looking up text with no matching ticket at all returns None, not an error"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        match = await repo.find_ready_by_normalized_text("совсем ничего нет")

        assert match is None


@pytest.mark.asyncio
async def test_mark_ready_updates_ticket_fields(empty_db, test_session_maker) -> None:
    """mark_ready() writes the full classification result and flips status to ready"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        created = await repo.create(
            Ticket(
                raw_text="хочу записаться на просмотр",
                normalized_text="хочу записаться на просмотр",
                status=TicketStatusEnum.PROCESSING,
            )
        )
        await session.commit()
        ticket_id = created.id

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        result = ClassificationResult(
            category=TicketCategoryEnum.VIEWING,
            summary="Клиент хочет записаться на просмотр",
            priority=TicketPriorityEnum.HIGH,
            entities={"date": "завтра"},
            ai_used=True,
            prompt_tokens=10,
            completion_tokens=20,
            llm_response_time_ms=500,
        )
        updated = await repo.mark_ready(ticket_id, result)
        await session.commit()

        assert updated.status == TicketStatusEnum.READY
        assert updated.category == TicketCategoryEnum.VIEWING
        assert updated.entities == {"date": "завтра"}
        assert updated.ai_used is True
        assert updated.prompt_tokens == 10
        assert updated.completion_tokens == 20
        assert updated.llm_response_time_ms == 500


@pytest.mark.asyncio
async def test_mark_failed_increments_retries_and_sets_error(
    empty_db, test_session_maker
) -> None:
    """mark_failed() flips status to failed, records the error, and bumps retries each call"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        created = await repo.create(
            Ticket(
                raw_text="что-то сломалось",
                normalized_text="что-то сломалось",
                status=TicketStatusEnum.PROCESSING,
            )
        )
        await session.commit()
        ticket_id = created.id

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        updated = await repo.mark_failed(ticket_id, "LLM timeout")
        await session.commit()

        assert updated.status == TicketStatusEnum.FAILED
        assert updated.retries == 1
        assert updated.error_message == "LLM timeout"

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        updated_again = await repo.mark_failed(ticket_id, "LLM timeout again")
        await session.commit()

        assert updated_again.retries == 2
        assert updated_again.error_message == "LLM timeout again"
