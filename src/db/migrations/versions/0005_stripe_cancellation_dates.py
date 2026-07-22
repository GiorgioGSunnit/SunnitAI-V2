"""Store Stripe subscription cancellation timestamps.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("tenant_subscriptions")}
    if "cancel_at" not in columns:
        op.add_column(
            "tenant_subscriptions",
            sa.Column("cancel_at", sa.DateTime(timezone=True), nullable=True),
        )
    if "canceled_at" not in columns:
        op.add_column(
            "tenant_subscriptions",
            sa.Column("canceled_at", sa.DateTime(timezone=True), nullable=True),
        )
    if "ended_at" not in columns:
        op.add_column(
            "tenant_subscriptions",
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("tenant_subscriptions", "ended_at")
    op.drop_column("tenant_subscriptions", "canceled_at")
    op.drop_column("tenant_subscriptions", "cancel_at")
