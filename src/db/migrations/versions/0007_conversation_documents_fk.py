"""Add the conversation foreign key now that malformed test sessions are gone.

The join table was created without this reference because ~370 development
sessions carried non-UUID ids, which blocked the conversations table from being
populated reliably. Those have been removed and the sync failure they caused is
fixed, so the constraint can now be enforced.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-10
"""

from alembic import op
from sqlalchemy import inspect

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

_FK = "fk_conversation_documents_conversation_id"


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if "conversation_documents" not in insp.get_table_names():
        return
    if any(fk["name"] == _FK for fk in insp.get_foreign_keys("conversation_documents")):
        return

    # Drop links pointing at conversations that never made it into Postgres,
    # otherwise the constraint cannot be created.
    op.execute(
        """
        DELETE FROM conversation_documents cd
        WHERE NOT EXISTS (
            SELECT 1 FROM conversations c WHERE c.id = cd.conversation_id
        )
        """
    )
    op.create_foreign_key(
        _FK,
        "conversation_documents",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(_FK, "conversation_documents", type_="foreignkey")
