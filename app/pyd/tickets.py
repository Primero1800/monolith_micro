from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.common.enums import TicketCategoryEnum, TicketPriorityEnum, TicketStatusEnum


class TicketAnalyzeRequest(BaseModel):
    """Input payload for synchronous ticket analysis"""

    text: str = Field(
        ...,
        max_length=500,
        examples=["Здравствуйте, хочу снять квартиру в центре, бюджет 40 тысяч"],
    )


class TicketAnalyzeResponse(BaseModel):
    """Structured result of synchronous ticket analysis"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    status: TicketStatusEnum
    category: TicketCategoryEnum | None = None
    summary: str | None = None
    priority: TicketPriorityEnum | None = None
    entities: dict[str, Any] | None = None
    ai_used: bool | None = None


class TicketAdminResponse(TicketAnalyzeResponse):
    """Extended ticket view for admins — public fields plus internal technical data"""

    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    llm_response_time_ms: int | None = None
    retries: int
    error_message: str | None = None
