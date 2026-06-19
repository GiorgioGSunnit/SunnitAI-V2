"""Platform tenant row + user_documents table.

Revision ID: 0001
Revises:
Create Date: 2026-06-18
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
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

    # Operation B — create user_documents table
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            original_filename TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            file_size_bytes INTEGER,
            uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_user_documents_tenant_user
            ON user_documents (tenant_id, user_id);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS user_documents;")
    op.execute("""
        DELETE FROM tenants WHERE id = '00000000-0000-0000-0000-000000000001';
    """)
