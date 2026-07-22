"""Add tenant_subscriptions table for Stripe billing state.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-26
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if not inspector.has_table("tenant_subscriptions"):
        op.create_table(
            "tenant_subscriptions",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("plan_id", sa.String(length=50), nullable=True),
            sa.Column("status", sa.String(length=50), nullable=False, server_default="inactive"),
            sa.Column("seats", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("stripe_customer_id", sa.String(length=255), nullable=True),
            sa.Column("stripe_subscription_id", sa.String(length=255), nullable=True),
            sa.Column("stripe_checkout_session_id", sa.String(length=255), nullable=True),
            sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=True),
            sa.Column("cancel_at_period_end", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("last_payment_status", sa.String(length=50), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("tenant_id"),
            sa.UniqueConstraint("stripe_subscription_id"),
            sa.UniqueConstraint("stripe_checkout_session_id"),
        )
        op.create_index(
            "ix_tenant_subscriptions_customer_id",
            "tenant_subscriptions",
            ["stripe_customer_id"],
            unique=False,
        )
    elif "ix_tenant_subscriptions_customer_id" not in {
        index["name"] for index in inspector.get_indexes("tenant_subscriptions")
    }:
        op.create_index(
            "ix_tenant_subscriptions_customer_id",
            "tenant_subscriptions",
            ["stripe_customer_id"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_index("ix_tenant_subscriptions_customer_id", table_name="tenant_subscriptions")
    op.drop_table("tenant_subscriptions")
