"""Add trial timestamps to tenant_subscriptions.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-26
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("tenant_subscriptions")}
    if "trial_started_at" not in columns:
        op.add_column("tenant_subscriptions", sa.Column("trial_started_at", sa.DateTime(timezone=True), nullable=True))
    if "trial_ends_at" not in columns:
        op.add_column("tenant_subscriptions", sa.Column("trial_ends_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("tenant_subscriptions", "trial_ends_at")
    op.drop_column("tenant_subscriptions", "trial_started_at")
