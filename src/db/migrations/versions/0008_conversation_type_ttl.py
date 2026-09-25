"""Add session_type and expires_at to conversations.

session_type distinguishes rag / calculation / document sessions so the
chat layer can size the LLM context window per type. expires_at gives
document sessions a TTL (set when a document is attached) so their
uploaded-file context doesn't outlive the upload itself; rag/calculation
sessions leave it NULL and are never swept.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-23
"""

from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("session_type", sa.String(20), nullable=False, server_default="rag"),
    )
    op.add_column(
        "conversations",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Belt-and-suspenders on top of server_default, for any row written
    # through a path that bypassed it.
    op.execute("UPDATE conversations SET session_type = 'rag'")
    op.create_index(
        "ix_conversations_expires_at",
        "conversations",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_conversations_expires_at", table_name="conversations")
    op.drop_column("conversations", "expires_at")
    op.drop_column("conversations", "session_type")
