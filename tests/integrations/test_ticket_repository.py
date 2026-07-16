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
async def test_get_by_id_returns_the_matching_ticket(
    empty_db, test_session_maker
) -> None:
    """get_by_id() fetches the ticket with the given id"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        created = await repo.create(
            Ticket(
                raw_text="хочу снять квартиру",
                normalized_text="хочу снять квартиру",
                status=TicketStatusEnum.DRAFT,
            )
        )
        await session.commit()
        ticket_id = created.id

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        found = await repo.get_by_id(ticket_id)

        assert found is not None
        assert found.id == ticket_id
        assert found.raw_text == "хочу снять квартиру"


@pytest.mark.asyncio
async def test_get_by_id_returns_none_when_missing(
    empty_db, test_session_maker
) -> None:
    """get_by_id() returns None for an id that doesn't exist, not an error"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        found = await repo.get_by_id(999999)

        assert found is None


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


@pytest.mark.asyncio
async def test_mark_failed_transitions_to_dead_letter_after_max_retries(
    empty_db, test_session_maker
) -> None:
    """mark_failed() moves the ticket to dead_letter once retries reach settings.MAX_RETRIES"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        created = await repo.create(
            Ticket(
                raw_text="постоянно падает",
                normalized_text="постоянно падает",
                status=TicketStatusEnum.PROCESSING,
            )
        )
        await session.commit()
        ticket_id = created.id

    updated = None
    for _ in range(3):
        async with test_session_maker() as session:
            repo = TicketRepository(session)
            updated = await repo.mark_failed(ticket_id, "LLM timeout")
            await session.commit()

    assert updated is not None
    assert updated.retries == 3
    assert updated.status == TicketStatusEnum.DEAD_LETTER


@pytest.mark.asyncio
async def test_claim_pending_picks_up_draft_failed_and_stuck_processing(
    empty_db, test_session_maker
) -> None:
    """claim_pending() picks up draft/failed tickets and processing ones stuck past the timeout"""
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import update as sa_update

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        draft = await repo.create(
            Ticket(
                raw_text="draft", normalized_text="draft", status=TicketStatusEnum.DRAFT
            )
        )
        failed = await repo.create(
            Ticket(
                raw_text="failed",
                normalized_text="failed",
                status=TicketStatusEnum.FAILED,
            )
        )
        stuck = await repo.create(
            Ticket(
                raw_text="stuck",
                normalized_text="stuck",
                status=TicketStatusEnum.PROCESSING,
            )
        )
        fresh = await repo.create(
            Ticket(
                raw_text="fresh",
                normalized_text="fresh",
                status=TicketStatusEnum.PROCESSING,
            )
        )
        ready = await repo.create(
            Ticket(
                raw_text="ready", normalized_text="ready", status=TicketStatusEnum.READY
            )
        )
        await session.flush()

        stale_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        await session.execute(
            sa_update(Ticket).where(Ticket.id == stuck.id).values(updated_at=stale_time)
        )
        await session.commit()

        draft_id, failed_id, stuck_id, fresh_id, ready_id = (
            draft.id,
            failed.id,
            stuck.id,
            fresh.id,
            ready.id,
        )

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        claimed = await repo.claim_pending(batch_size=10, stuck_after_sec=30)
        await session.commit()

        claimed_ids = {t.id for t in claimed}
        assert claimed_ids == {draft_id, failed_id, stuck_id}
        assert fresh_id not in claimed_ids
        assert ready_id not in claimed_ids
        assert all(t.status == TicketStatusEnum.PROCESSING for t in claimed)


@pytest.mark.asyncio
async def test_claim_pending_respects_batch_size_oldest_first(
    empty_db, test_session_maker
) -> None:
    """claim_pending() claims at most batch_size tickets, oldest (by created_at) first"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        created_ids = []
        for i in range(3):
            ticket = await repo.create(
                Ticket(
                    raw_text=f"draft {i}",
                    normalized_text=f"draft {i}",
                    status=TicketStatusEnum.DRAFT,
                )
            )
            await session.flush()
            created_ids.append(ticket.id)
        await session.commit()

    async with test_session_maker() as session:
        repo = TicketRepository(session)
        claimed = await repo.claim_pending(batch_size=2, stuck_after_sec=30)
        await session.commit()

        assert len(claimed) == 2
        assert [t.id for t in claimed] == created_ids[:2]


@pytest.mark.asyncio
async def test_claim_pending_returns_empty_when_nothing_matches(
    empty_db, test_session_maker
) -> None:
    """claim_pending() returns an empty sequence when there's nothing to pick up"""
    async with test_session_maker() as session:
        repo = TicketRepository(session)
        claimed = await repo.claim_pending(batch_size=10, stuck_after_sec=30)

        assert list(claimed) == []
