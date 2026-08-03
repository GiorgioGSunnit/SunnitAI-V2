"""Record how each quoted premium was arrived at.

Adds two columns to ``pc_normalized_quotes``:

``calculation_source``
    ``demonstration_formula`` when this application computed the price,
    ``provider_supplied`` when an insurer quoted it. Existing rows are
    backfilled from ``is_demonstration``, which is the same distinction those
    rows already carried.

``calculation_breakdown``
    The auditable step-by-step derivation, JSON, demonstration quotes only.
    Stored rather than recomputed so the breakdown a user was shown remains
    reproducible even after the formula changes.

A separate revision rather than an edit to 0001: that migration has already
been applied, so rewriting it would leave existing databases silently out of
step with the migration history.

Revision ID: 0002_calculation_breakdown
Revises: 0001_initial
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

import policy_comparator.db


revision = "0002_calculation_breakdown"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Batch mode: SQLite cannot ALTER a table in place, and local development
    # runs on SQLite. On PostgreSQL this compiles to a plain ALTER TABLE.
    with op.batch_alter_table("pc_normalized_quotes") as batch:
        batch.add_column(
            sa.Column(
                "calculation_source",
                sa.String(length=32),
                nullable=False,
                server_default="provider_supplied",
            )
        )
        batch.add_column(
            sa.Column(
                "calculation_breakdown",
                policy_comparator.db.JSONColumn(),
                nullable=True,
            )
        )

    # Existing demonstration quotes predate the breakdown, so they get the
    # correct source but no steps — inventing a derivation for a price that was
    # computed by an older formula would be a fabrication.
    op.execute(
        "UPDATE pc_normalized_quotes "
        "SET calculation_source = 'demonstration_formula' "
        "WHERE is_demonstration = true"
    )


def downgrade() -> None:
    with op.batch_alter_table("pc_normalized_quotes") as batch:
        batch.drop_column("calculation_breakdown")
        batch.drop_column("calculation_source")
