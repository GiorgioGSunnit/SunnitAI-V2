"""Add formula registry tables (formulas + istat_coefficients) with pgvector.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-26

Execution order (must not be reordered):
  1. CREATE EXTENSION IF NOT EXISTS vector
  2. CREATE TABLE formulas
  3. CREATE TABLE istat_coefficients
  4. B-tree indexes on formulas(category) and formulas(slug)
  5. IVFFlat vector index on formulas(embedding)

downgrade() drops both tables but leaves the vector extension in place
to avoid breaking other potential consumers of pgvector on the same DB.
"""
from alembic import op


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Step 1: enable pgvector extension ─────────────────────────────────────
    # Must come BEFORE any DDL that uses the vector type.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # ── Step 2: formulas table ────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE formulas (
            id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            slug             VARCHAR(120) UNIQUE NOT NULL,
            name_it          VARCHAR(255) NOT NULL,
            description_it   TEXT         NOT NULL,
            category         VARCHAR(80)  NOT NULL,
            subcategory      VARCHAR(80),
            expression_type  VARCHAR(20)  NOT NULL
                                 CHECK (expression_type IN ('simple', 'complex')),
            expression       TEXT,
            plugin_name      VARCHAR(120),
            parameter_schema JSONB        NOT NULL,
            source_norm      VARCHAR(255),
            examples         JSONB,
            embedding        vector(1024),
            is_active        BOOLEAN      NOT NULL DEFAULT TRUE,
            created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
        );
    """)

    # ── Step 3: istat_coefficients table ──────────────────────────────────────
    op.execute("""
        CREATE TABLE istat_coefficients (
            year        INTEGER         PRIMARY KEY,
            cpi_annual  NUMERIC(6, 4)   NOT NULL,
            tfr_coeff   NUMERIC(7, 4)   NOT NULL
        );
    """)

    # ── Step 4: B-tree indexes ────────────────────────────────────────────────
    op.execute(
        "CREATE INDEX formulas_category_idx ON formulas (category);"
    )
    op.execute(
        "CREATE INDEX formulas_slug_idx ON formulas (slug);"
    )

    # ── Step 5: IVFFlat vector index ──────────────────────────────────────────
    # lists=10 is appropriate for a small POC dataset (4 formulas).
    # For production with hundreds of formulas, raise to sqrt(n_rows).
    # Requires at least one row in the table before the index can be built —
    # if the table is empty at migration time, the index is still created but
    # will be vacuumed/populated on first insert.
    op.execute("""
        CREATE INDEX formulas_embedding_idx ON formulas
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10);
    """)


def downgrade() -> None:
    # Drop indexes first (implicit via DROP TABLE, but explicit is safer)
    op.execute("DROP TABLE IF EXISTS istat_coefficients;")
    op.execute("DROP TABLE IF EXISTS formulas;")
    # NOTE: vector extension is intentionally NOT dropped here.
    # Dropping it would break other tables/indexes that may use it.
    # If you need to remove it: DROP EXTENSION vector CASCADE; (manual step)
