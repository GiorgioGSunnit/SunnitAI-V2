"""Add anchor accumulation and rolling summary to conversations.

anchors carries small, deduplicated references extracted from RAG turns
(document ids cited, article refs, law hints, CCNL sector, calculator types)
so a follow-up question can be grounded even after older turns are
summarized away. summary + summary_covers_turns back the rolling
compression in ChatSession._summarize_if_needed(): summary holds the LLM's
running recap of the oldest messages, and summary_covers_turns counts how
many original messages that recap has absorbed so far.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column(
            "anchors", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
    )
    op.add_column(
        "conversations",
        sa.Column("summary", sa.Text(), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("summary_covers_turns", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("conversations", "summary_covers_turns")
    op.drop_column("conversations", "summary")
    op.drop_column("conversations", "anchors")
