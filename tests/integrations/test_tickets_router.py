import json
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from app.adapters.ai_client import ChatResult
from app.common.exceptions import IntegrityDataException
from app.dependencies.infrastructure import get_ai_client
from app.main import app
from app.services.ticket_service import TicketService


def _override_ai_client(fake_client: AsyncMock) -> None:
    """Swap the app's real AI client dependency for the given fake, for this test's requests"""
    app.dependency_overrides[get_ai_client] = lambda: fake_client


def _llm_success(category: str, summary: str, priority: str) -> ChatResult:
    """Build a ChatResult whose content is a valid LLMClassificationOutput JSON payload"""
    return ChatResult(
        content=json.dumps(
            {"category": category, "summary": summary, "priority": priority}
        ),
        prompt_tokens=42,
        completion_tokens=17,
        total_tokens=59,
    )


@pytest.mark.asyncio
async def test_analyze_degenerate_text_returns_other_without_llm(
    empty_db, async_client: AsyncClient
) -> None:
    """Punctuation-only input is classified as "other" over the real DB, without calling the LLM"""
    fake_ai_client = AsyncMock()
    _override_ai_client(fake_ai_client)

    response = await async_client.post(
        "/api/v1/tickets/analyze", json={"text": "   !!! ..."}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["category"] == "other"
    assert body["ai_used"] is False
    fake_ai_client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_new_text_calls_llm_and_returns_structured_result(
    empty_db, async_client: AsyncClient
) -> None:
    """A fresh, long ticket text (no regex fast-path match) is classified via the (mocked) LLM"""
    fake_ai_client = AsyncMock()
    fake_ai_client.chat.return_value = _llm_success(
        "rent", "Клиент хочет снять квартиру", "medium"
    )
    _override_ai_client(fake_ai_client)

    response = await async_client.post(
        "/api/v1/tickets/analyze",
        json={
            "text": "Здравствуйте, расскажите пожалуйста о ваших актуальных "
            "предложениях по недвижимости в центре города"
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["category"] == "rent"
    assert body["summary"] == "Клиент хочет снять квартиру"
    assert body["ai_used"] is True
    fake_ai_client.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_duplicate_request_skips_second_llm_call(
    empty_db, async_client: AsyncClient
) -> None:
    """Submitting the same text twice reuses the first ticket's result and skips a second LLM call"""
    fake_ai_client = AsyncMock()
    fake_ai_client.chat.return_value = _llm_success(
        "sale", "Клиент хочет продать квартиру", "high"
    )
    _override_ai_client(fake_ai_client)

    text = {
        "text": "Добрый день, хотелось бы узнать больше о вашей компании "
        "и вариантах, которые вы можете предложить"
    }
    first = await async_client.post("/api/v1/tickets/analyze", json=text)
    second = await async_client.post("/api/v1/tickets/analyze", json=text)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["category"] == second.json()["category"] == "sale"
    assert second.json()["ai_used"] is False
    assert first.json()["id"] != second.json()["id"]
    fake_ai_client.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_short_unambiguous_text_uses_regex_fast_path(
    empty_db, async_client: AsyncClient
) -> None:
    """Short text with exactly one category keyword is classified without any LLM call"""
    fake_ai_client = AsyncMock()
    _override_ai_client(fake_ai_client)

    response = await async_client.post(
        "/api/v1/tickets/analyze", json={"text": "хочу снять квартиру"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["category"] == "rent"
    assert body["ai_used"] is False
    fake_ai_client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_llm_failure_marks_ticket_failed_but_returns_200(
    empty_db, async_client: AsyncClient
) -> None:
    """An LLM call with no content marks the ticket failed, but the HTTP response is still 200"""
    fake_ai_client = AsyncMock()
    fake_ai_client.chat.return_value = ChatResult(
        content=None, prompt_tokens=0, completion_tokens=0, total_tokens=0
    )
    _override_ai_client(fake_ai_client)

    response = await async_client.post(
        "/api/v1/tickets/analyze",
        json={"text": "Что-то про недвижимость без ответа LLM"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["category"] is None


@pytest.mark.asyncio
async def test_analyze_llm_invalid_json_marks_ticket_failed(
    empty_db, async_client: AsyncClient
) -> None:
    """LLM content that isn't valid classification JSON also marks the ticket failed"""
    fake_ai_client = AsyncMock()
    fake_ai_client.chat.return_value = ChatResult(
        content="not a json object at all",
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
    )
    _override_ai_client(fake_ai_client)

    response = await async_client.post(
        "/api/v1/tickets/analyze",
        json={"text": "Ещё одно сообщение без валидного JSON от LLM"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"


@pytest.mark.asyncio
async def test_analyze_rejects_missing_text_field(
    empty_db, async_client: AsyncClient
) -> None:
    """A request body without the required text field is rejected with 422, before hitting the service"""
    response = await async_client.post("/api/v1/tickets/analyze", json={})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_translates_integrity_data_exception_to_http_error(
    empty_db, async_client: AsyncClient, mocker
) -> None:
    """A genuine DB integrity error from the service surfaces as its mapped HTTP status/detail"""
    mocker.patch.object(
        TicketService,
        "analyze",
        side_effect=IntegrityDataException(
            detail="Key (id)=(1) already exists.",
            status_code=409,
            headers={"X-Reason": "duplicate"},
        ),
    )

    response = await async_client.post(
        "/api/v1/tickets/analyze", json={"text": "любой текст"}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Key (id)=(1) already exists."
    assert response.headers["x-reason"] == "duplicate"


@pytest.mark.asyncio
async def test_create_ticket_returns_202_with_draft_status(
    empty_db, async_client: AsyncClient
) -> None:
    """POST /tickets queues a draft ticket immediately, without classifying or calling the LLM"""
    fake_ai_client = AsyncMock()
    _override_ai_client(fake_ai_client)

    response = await async_client.post(
        "/api/v1/tickets", json={"text": "хочу снять квартиру"}
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "draft"
    assert set(body.keys()) == {"id", "status"}
    fake_ai_client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_create_ticket_translates_integrity_data_exception_to_http_error(
    empty_db, async_client: AsyncClient, mocker
) -> None:
    """POST /tickets maps a DB integrity error the same way /analyze does"""
    mocker.patch.object(
        TicketService,
        "create_draft",
        side_effect=IntegrityDataException(
            detail="Key (id)=(1) already exists.",
            status_code=409,
            headers={"X-Reason": "duplicate"},
        ),
    )

    response = await async_client.post("/api/v1/tickets", json={"text": "любой текст"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Key (id)=(1) already exists."
    assert response.headers["x-reason"] == "duplicate"


@pytest.mark.asyncio
async def test_create_ticket_rejects_missing_text_field(
    empty_db, async_client: AsyncClient
) -> None:
    """POST /tickets validates the payload the same way /analyze does"""
    response = await async_client.post("/api/v1/tickets", json={})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_ticket_returns_the_previously_created_draft(
    empty_db, async_client: AsyncClient
) -> None:
    """GET /tickets/{id} returns the same ticket a prior POST /tickets created"""
    created = await async_client.post(
        "/api/v1/tickets", json={"text": "хочу снять квартиру"}
    )
    ticket_id = created.json()["id"]

    response = await async_client.get(f"/api/v1/tickets/{ticket_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == ticket_id
    assert body["status"] == "draft"


@pytest.mark.asyncio
async def test_get_ticket_not_found_returns_404(
    empty_db, async_client: AsyncClient
) -> None:
    """GET /tickets/{id} returns 404 for an id that doesn't exist"""
    response = await async_client.get("/api/v1/tickets/999999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket not found"


@pytest.mark.asyncio
async def test_get_ticket_public_response_omits_technical_fields(
    empty_db, async_client: AsyncClient
) -> None:
    """The public GET /tickets/{id} response never leaks internal technical fields"""
    fake_ai_client = AsyncMock()
    fake_ai_client.chat.return_value = _llm_success(
        "rent", "Клиент хочет снять квартиру", "medium"
    )
    _override_ai_client(fake_ai_client)
    created = await async_client.post(
        "/api/v1/tickets/analyze",
        json={
            "text": "Здравствуйте, расскажите пожалуйста о ваших актуальных "
            "предложениях по недвижимости в центре города"
        },
    )
    ticket_id = created.json()["id"]

    response = await async_client.get(f"/api/v1/tickets/{ticket_id}")

    assert response.status_code == 200
    body = response.json()
    for technical_field in (
        "raw_text",
        "prompt_tokens",
        "completion_tokens",
        "llm_response_time_ms",
        "retries",
        "error_message",
    ):
        assert technical_field not in body


@pytest.mark.asyncio
async def test_get_ticket_admin_returns_full_record_with_technical_fields(
    empty_db, async_client: AsyncClient
) -> None:
    """The admin GET endpoint exposes raw_text and LLM technical metadata the public one hides"""
    fake_ai_client = AsyncMock()
    fake_ai_client.chat.return_value = _llm_success(
        "rent", "Клиент хочет снять квартиру", "medium"
    )
    _override_ai_client(fake_ai_client)
    raw_text = (
        "Здравствуйте, расскажите пожалуйста о ваших актуальных "
        "предложениях по недвижимости в центре города"
    )
    created = await async_client.post(
        "/api/v1/tickets/analyze", json={"text": raw_text}
    )
    ticket_id = created.json()["id"]

    response = await async_client.get(f"/api/v1/admin/tickets/{ticket_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["raw_text"] == raw_text
    assert body["prompt_tokens"] == 42
    assert body["completion_tokens"] == 17
    assert body["llm_response_time_ms"] is not None
    assert body["retries"] == 0
    assert body["error_message"] is None


@pytest.mark.asyncio
async def test_get_ticket_admin_not_found_returns_404(
    empty_db, async_client: AsyncClient
) -> None:
    """GET /admin/tickets/{id} returns 404 for an id that doesn't exist, same as the public route"""
    response = await async_client.get("/api/v1/admin/tickets/999999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket not found"
