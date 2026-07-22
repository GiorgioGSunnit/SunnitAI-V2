"""Add superadmin access overrides to tenant subscriptions.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("tenant_subscriptions")}
    if "admin_access_override" not in columns:
        op.add_column(
            "tenant_subscriptions",
            sa.Column("admin_access_override", sa.String(length=20), nullable=True),
        )
    if "admin_access_until" not in columns:
        op.add_column(
            "tenant_subscriptions",
            sa.Column("admin_access_until", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("tenant_subscriptions", "admin_access_until")
    op.drop_column("tenant_subscriptions", "admin_access_override")
