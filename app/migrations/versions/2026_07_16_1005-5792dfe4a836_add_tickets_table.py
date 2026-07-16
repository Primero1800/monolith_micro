"""add tickets table

Revision ID: 5792dfe4a836
Revises:
Create Date: 2026-07-16 10:05:02.644039

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "5792dfe4a836"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_STATUS_ENUM_NAME = "ticketstatusenum"
_STATUS_ENUM_VALUES = ("DRAFT", "PROCESSING", "READY", "FAILED", "DEAD_LETTER")

_CATEGORY_ENUM_NAME = "ticketcategoryenum"
_CATEGORY_ENUM_VALUES = (
    "RENT",
    "SALE",
    "VIEWING",
    "CONSULTATION",
    "COMPLAINT",
    "OTHER",
)

_PRIORITY_ENUM_NAME = "ticketpriorityenum"
_PRIORITY_ENUM_VALUES = ("LOW", "MEDIUM", "HIGH")


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "tickets",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
            comment="Уникальный идентификатор",
        ),
        sa.Column(
            "raw_text",
            sa.Text(),
            nullable=False,
            comment="Исходный текст обращения",
        ),
        sa.Column(
            "status",
            sa.Enum(*_STATUS_ENUM_VALUES, name=_STATUS_ENUM_NAME),
            server_default=sa.text("'DRAFT'"),
            nullable=False,
            comment="Статус обработки тикета",
        ),
        sa.Column(
            "category",
            sa.Enum(*_CATEGORY_ENUM_VALUES, name=_CATEGORY_ENUM_NAME),
            nullable=True,
            comment="Категория обращения",
        ),
        sa.Column(
            "summary",
            sa.Text(),
            nullable=True,
            comment="Краткое саммари обращения",
        ),
        sa.Column(
            "priority",
            sa.Enum(*_PRIORITY_ENUM_VALUES, name=_PRIORITY_ENUM_NAME),
            nullable=True,
            comment="Приоритет обращения",
        ),
        sa.Column(
            "entities",
            JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Извлечённые сущности (телефон, бюджет, район и т.п.)",
        ),
        sa.Column(
            "ai_used",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
            comment="Использовалась ли LLM для классификации",
        ),
        sa.Column(
            "tokens_used",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
            comment="Суммарное число токенов, потраченных на обработку",
        ),
        sa.Column(
            "processing_time_ms",
            sa.Integer(),
            nullable=True,
            comment="Время обработки в миллисекундах",
        ),
        sa.Column(
            "retries",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
            comment="Количество попыток обработки",
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
            comment="Текст последней ошибки обработки",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("TIMEZONE('utc', now())"),
            nullable=False,
            comment="Время создания записи",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("TIMEZONE('utc', now())"),
            nullable=False,
            comment="Время обновления записи",
        ),
        sa.PrimaryKeyConstraint("id"),
        comment="Обращения в поддержку — вход, классификация и результат",
    )
    op.create_index(
        "ix_tickets_status_created_at",
        "tickets",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_tickets_status_created_at", table_name="tickets")
    op.drop_table("tickets")
    sa.Enum(*_STATUS_ENUM_VALUES, name=_STATUS_ENUM_NAME).drop(op.get_bind())
    sa.Enum(*_CATEGORY_ENUM_VALUES, name=_CATEGORY_ENUM_NAME).drop(op.get_bind())
    sa.Enum(*_PRIORITY_ENUM_VALUES, name=_PRIORITY_ENUM_NAME).drop(op.get_bind())
