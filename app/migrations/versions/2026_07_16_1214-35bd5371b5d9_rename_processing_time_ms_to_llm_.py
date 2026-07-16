"""rename processing_time_ms to llm_response_time_ms

Revision ID: 35bd5371b5d9
Revises: d234ba37270d
Create Date: 2026-07-16 12:14:39.643706

"""

from typing import Sequence, Union

from alembic import op

revision: str = "35bd5371b5d9"
down_revision: Union[str, Sequence[str], None] = "d234ba37270d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "tickets",
        "processing_time_ms",
        new_column_name="llm_response_time_ms",
        comment="Время именно LLM-запроса в миллисекундах (не всего тикета)",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "tickets",
        "llm_response_time_ms",
        new_column_name="processing_time_ms",
        comment="Время обработки в миллисекундах",
    )
