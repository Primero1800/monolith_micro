from dataclasses import dataclass
from typing import Any

from sqlalchemy import Enum as SqlEnum
from sqlalchemy import Index, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import TicketCategoryEnum, TicketPriorityEnum, TicketStatusEnum
from app.models.base import Base, int_pk


@dataclass
class ClassificationResult:
    """Outcome of classifying a ticket, regardless of which pipeline step produced it"""

    category: TicketCategoryEnum
    summary: str
    priority: TicketPriorityEnum
    entities: dict[str, Any] | None
    ai_used: bool


class Ticket(Base):
    """SQLAlchemy model for a support ticket submitted to the AI agent"""

    __tablename__ = "tickets"
    __table_args__ = (
        Index("ix_tickets_status_created_at", "status", "created_at"),
        Index("ix_tickets_normalized_text", "normalized_text"),
        {"comment": "Обращения в поддержку — вход, классификация и результат"},
    )

    id: Mapped[int_pk]
    raw_text: Mapped[str] = mapped_column(Text, comment="Исходный текст обращения")
    normalized_text: Mapped[str] = mapped_column(
        Text,
        comment="raw_text в нижнем регистре без пунктуации — для поиска дублей",
    )
    status: Mapped[TicketStatusEnum] = mapped_column(
        SqlEnum(TicketStatusEnum),
        default=TicketStatusEnum.DRAFT,
        server_default=text(f"'{TicketStatusEnum.DRAFT.name}'"),
        comment="Статус обработки тикета",
    )
    category: Mapped[TicketCategoryEnum | None] = mapped_column(
        SqlEnum(TicketCategoryEnum), comment="Категория обращения"
    )
    summary: Mapped[str | None] = mapped_column(
        Text, comment="Краткое саммари обращения"
    )
    priority: Mapped[TicketPriorityEnum | None] = mapped_column(
        SqlEnum(TicketPriorityEnum), comment="Приоритет обращения"
    )
    entities: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, comment="Извлечённые сущности (телефон, бюджет, район и т.п.)"
    )
    ai_used: Mapped[bool] = mapped_column(
        default=False,
        server_default=text("false"),
        comment="Использовалась ли LLM для классификации",
    )
    tokens_used: Mapped[int] = mapped_column(
        default=0,
        server_default=text("0"),
        comment="Суммарное число токенов, потраченных на обработку",
    )
    processing_time_ms: Mapped[int | None] = mapped_column(
        comment="Время обработки в миллисекундах"
    )
    retries: Mapped[int] = mapped_column(
        default=0,
        server_default=text("0"),
        comment="Количество попыток обработки",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, comment="Текст последней ошибки обработки"
    )
