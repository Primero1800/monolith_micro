from typing import Any

from pydantic import BaseModel, Field

from app.common.enums import TicketCategoryEnum, TicketPriorityEnum, TicketStatusEnum


class TicketAnalyzeRequest(BaseModel):
    """Input payload for synchronous ticket analysis"""

    text: str = Field(
        ...,
        examples=["Здравствуйте, хочу снять квартиру в центре, бюджет 40 тысяч"],
    )


class TicketAnalyzeResponse(BaseModel):
    """Structured result of synchronous ticket analysis"""

    id: int
    status: TicketStatusEnum
    category: TicketCategoryEnum | None = None
    summary: str | None = None
    priority: TicketPriorityEnum | None = None
    entities: dict[str, Any] | None = None
    ai_used: bool | None = None
    message: str | None = None
