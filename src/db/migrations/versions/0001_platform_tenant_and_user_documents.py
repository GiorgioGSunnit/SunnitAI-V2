"""Platform tenant row + user_documents table.

Revision ID: 0001
Revises: 8eca0a3dd1e6
Create Date: 2026-06-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0001"
down_revision = "8eca0a3dd1e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Operation A — insert platform tenant
    op.execute("""
        INSERT INTO tenants (id, email, hashed_password, plan, is_active, created_at, updated_at)
        VALUES (
            '00000000-0000-0000-0000-000000000001',
            'platform@sunnit.internal',
            'LOCKED',
            'platform',
            true,
            NOW(),
            NOW()
        )
        ON CONFLICT (id) DO NOTHING;
    """)

    op.execute("""
        INSERT INTO tenant_profiles (id, tenant_id, display_name)
        VALUES (gen_random_uuid(), '00000000-0000-0000-0000-000000000001', 'Sunnit Platform')
        ON CONFLICT DO NOTHING;
    """)

    inspector = inspect(op.get_bind())
    if not inspector.has_table("user_documents"):
        op.create_table(
            "user_documents",
            sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("tenant_id", sa.UUID(), nullable=False),
            sa.Column("original_filename", sa.String(length=255), nullable=False),
            sa.Column("storage_path", sa.Text(), nullable=False),
            sa.Column("file_size_bytes", sa.Integer(), nullable=True),
            sa.Column("scope", sa.String(length=20), nullable=False, server_default="personal"),
            sa.Column("document_role", sa.String(length=10), nullable=False, server_default="document"),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_user_documents_tenant_user",
            "user_documents",
            ["tenant_id", "user_id"],
        )
    elif "ix_user_documents_tenant_user" not in {
        index["name"] for index in inspector.get_indexes("user_documents")
    }:
        op.create_index(
            "ix_user_documents_tenant_user",
            "user_documents",
            ["tenant_id", "user_id"],
        )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS user_documents;")
    op.execute("""
        DELETE FROM tenants WHERE id = '00000000-0000-0000-0000-000000000001';
    """)
