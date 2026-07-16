from typing import Any

from pydantic import BaseModel

from app.common.enums import TicketCategoryEnum, TicketPriorityEnum


class LLMClassificationOutput(BaseModel):
    """Expected JSON shape of the LLM's ticket classification response"""

    category: TicketCategoryEnum
    summary: str
    priority: TicketPriorityEnum
    entities: dict[str, Any] | None = None
