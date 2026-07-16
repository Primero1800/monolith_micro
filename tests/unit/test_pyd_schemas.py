import pytest
from pydantic import ValidationError

from app.common.enums import TicketCategoryEnum, TicketPriorityEnum, TicketStatusEnum
from app.pyd.llm_classification import LLMClassificationOutput
from app.pyd.responses import HTTPExceptionResponse
from app.pyd.tickets import TicketAnalyzeRequest, TicketAnalyzeResponse


def test_ticket_analyze_request_requires_text() -> None:
    """text is a required field on the sync-analyze request payload"""
    with pytest.raises(ValidationError):
        TicketAnalyzeRequest()  # type: ignore[call-arg]

    request = TicketAnalyzeRequest(text="хочу снять квартиру")
    assert request.text == "хочу снять квартиру"


def test_ticket_analyze_response_defaults() -> None:
    """All classification fields default to None until a ticket is actually classified"""
    response = TicketAnalyzeResponse(id=1, status=TicketStatusEnum.PROCESSING)

    assert response.category is None
    assert response.summary is None
    assert response.priority is None
    assert response.entities is None
    assert response.ai_used is None
    assert response.message is None


def test_ticket_analyze_response_from_attributes() -> None:
    """The response schema builds from an ORM-like object via from_attributes"""

    class FakeTicket:
        """Minimal object shape matching the attributes TicketAnalyzeResponse reads"""

        id = 42
        status = TicketStatusEnum.READY
        category = TicketCategoryEnum.RENT
        summary = "Клиент хочет снять квартиру"
        priority = TicketPriorityEnum.MEDIUM
        entities = {"budget": "40000"}
        ai_used = True
        message = None

    response = TicketAnalyzeResponse.model_validate(FakeTicket())
    assert response.id == 42
    assert response.category == TicketCategoryEnum.RENT


def test_llm_classification_output_valid() -> None:
    """A well-formed LLM JSON payload parses into the expected typed fields"""
    parsed = LLMClassificationOutput.model_validate(
        {
            "category": "rent",
            "summary": "Клиент хочет снять квартиру",
            "priority": "medium",
            "entities": {"budget": "40000"},
        }
    )
    assert parsed.category == TicketCategoryEnum.RENT
    assert parsed.entities == {"budget": "40000"}


def test_llm_classification_output_entities_optional() -> None:
    """entities may be omitted entirely and defaults to None"""
    parsed = LLMClassificationOutput.model_validate(
        {"category": "other", "summary": "не по теме", "priority": "low"}
    )
    assert parsed.entities is None


@pytest.mark.parametrize(
    "payload",
    [
        {"category": "not_a_category", "summary": "x", "priority": "low"},
        {"category": "rent", "summary": "x", "priority": "not_a_priority"},
        {"category": "rent", "priority": "low"},
    ],
)
def test_llm_classification_output_rejects_invalid_payload(payload: dict) -> None:
    """Unknown enum values or missing required fields fail validation"""
    with pytest.raises(ValidationError):
        LLMClassificationOutput.model_validate(payload)


def test_http_exception_response_headers_optional() -> None:
    """headers defaults to None so exceptions without custom headers still validate"""
    response = HTTPExceptionResponse(detail="oops", status_code=503)
    assert response.headers is None
