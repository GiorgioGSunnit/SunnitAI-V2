"""Link documents to the conversations (cases) they belong to.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-10
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "conversation_documents" in inspect(op.get_bind()).get_table_names():
        return

    op.create_table(
        "conversation_documents",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "linked_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # conversation_id carries no foreign key on purpose. Conversations live
        # primarily in the sessions JSON file and are mirrored into Postgres on a
        # best-effort basis, so a row may not exist yet (or at all) when a
        # document is first used. A hard reference would silently drop exactly
        # the links this table exists to record.
        sa.ForeignKeyConstraint(
            ["document_id"], ["user_documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("conversation_id", "document_id"),
    )
    # "which documents belong to this case" is the common read; the reverse
    # ("which cases use this document") needs its own index.
    op.create_index(
        "ix_conversation_documents_document_id",
        "conversation_documents",
        ["document_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_documents_document_id",
        table_name="conversation_documents",
    )
    op.drop_table("conversation_documents")
