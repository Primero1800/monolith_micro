import json
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from app.adapters.ai_client import ChatResult
from app.dependencies.infrastructure import get_ai_client
from app.main import app


def _override_ai_client(fake_client: AsyncMock) -> None:
    app.dependency_overrides[get_ai_client] = lambda: fake_client


def _llm_success(category: str, summary: str, priority: str) -> ChatResult:
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
    fake_ai_client = AsyncMock()
    fake_ai_client.chat.return_value = _llm_success(
        "rent", "Клиент хочет снять квартиру", "medium"
    )
    _override_ai_client(fake_ai_client)

    response = await async_client.post(
        "/api/v1/tickets/analyze",
        json={"text": "Здравствуйте, хочу снять квартиру в центре"},
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
    fake_ai_client = AsyncMock()
    fake_ai_client.chat.return_value = _llm_success(
        "sale", "Клиент хочет продать квартиру", "high"
    )
    _override_ai_client(fake_ai_client)

    text = {"text": "Хочу продать квартиру в центре"}
    first = await async_client.post("/api/v1/tickets/analyze", json=text)
    second = await async_client.post("/api/v1/tickets/analyze", json=text)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["category"] == second.json()["category"] == "sale"
    assert second.json()["ai_used"] is False
    assert first.json()["id"] != second.json()["id"]
    fake_ai_client.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_llm_failure_marks_ticket_failed_but_returns_200(
    empty_db, async_client: AsyncClient
) -> None:
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
    response = await async_client.post("/api/v1/tickets/analyze", json={})

    assert response.status_code == 422
