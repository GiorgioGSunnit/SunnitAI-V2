"""Alembic migrations: they apply, they match the models, and they keep data.

These run Alembic in a subprocess against a throwaway SQLite file. The
application memoizes its settings and engine on first use, so driving Alembic
in-process would either pick up the test database or poison it for later tests.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALEMBIC_INI = PROJECT_ROOT / "policy_comparator" / "alembic.ini"


def alembic(*args: str, database_url: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PC_DATABASE_URL": database_url, "PC_MODE": "development"}
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode == 0, f"alembic {' '.join(args)} failed:\n{result.stderr}"
    return result


@pytest.fixture
def database(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'migration.db'}"


def columns(path: str, table: str) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    finally:
        connection.close()


class TestMigrations:
    def test_upgrade_head_creates_every_table(self, database, tmp_path):
        alembic("upgrade", "head", database_url=database)

        connection = sqlite3.connect(tmp_path / "migration.db")
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            connection.close()

        assert "alembic_version" in tables
        assert len([t for t in tables if t.startswith("pc_")]) == 16

    def test_head_is_the_calculation_revision(self, database):
        result = alembic("current", database_url=database)
        alembic("upgrade", "head", database_url=database)
        result = alembic("current", database_url=database)
        assert "0002_calculation_breakdown" in result.stdout

    def test_calculation_columns_are_added(self, database, tmp_path):
        alembic("upgrade", "head", database_url=database)
        present = columns(str(tmp_path / "migration.db"), "pc_normalized_quotes")
        assert "calculation_source" in present
        assert "calculation_breakdown" in present

    def test_the_second_revision_is_reversible(self, database, tmp_path):
        alembic("upgrade", "head", database_url=database)
        alembic("downgrade", "0001_initial", database_url=database)

        present = columns(str(tmp_path / "migration.db"), "pc_normalized_quotes")
        assert "calculation_source" not in present
        assert "calculation_breakdown" not in present

        alembic("upgrade", "head", database_url=database)
        assert "calculation_source" in columns(
            str(tmp_path / "migration.db"), "pc_normalized_quotes"
        )

    def test_existing_rows_are_backfilled_not_lost(self, database, tmp_path):
        """A database already carrying quotes must survive the upgrade."""
        alembic("upgrade", "0001_initial", database_url=database)

        path = str(tmp_path / "migration.db")
        connection = sqlite3.connect(path)
        tenant, customer, request, attempt, demo_quote, live_quote = (
            str(uuid.uuid4()) for _ in range(6)
        )
        # Column defaults are declared in Python, not in the schema, so raw SQL
        # has to supply every NOT NULL column itself.
        now = "2026-07-30 12:00:00"
        try:
            connection.execute(
                "INSERT INTO pc_customers "
                "(id, tenant_id, email, email_fingerprint, created_at, updated_at) "
                "VALUES (?, ?, 'x', 'fp', ?, ?)",
                (customer, tenant, now, now),
            )
            connection.execute(
                "INSERT INTO pc_quote_requests "
                "(id, tenant_id, customer_id, customer_profile_id, vehicle_id, "
                " insurance_history_id, coverage_preference_id, policy_start_date, "
                " selected_provider_ids, status, demonstration_data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, '2026-09-01', '[\"zurich\"]', 'completed', 1, ?, ?)",
                (request, tenant, customer, str(uuid.uuid4()), str(uuid.uuid4()),
                 str(uuid.uuid4()), str(uuid.uuid4()), now, now),
            )
            connection.execute(
                "INSERT INTO pc_provider_attempts "
                "(id, tenant_id, quote_request_id, provider_id, provider_type, provider_mode, "
                " status, attempt_count, idempotency_key, created_at, updated_at) "
                "VALUES (?, ?, ?, 'zurich', 'insurer', 'mock', 'quoted', 0, 'k', ?, ?)",
                (attempt, tenant, request, now, now),
            )
            for quote_id, is_demo in ((demo_quote, 1), (live_quote, 0)):
                connection.execute(
                    "INSERT INTO pc_normalized_quotes "
                    "(id, tenant_id, quote_request_id, provider_attempt_id, provider_id, "
                    " insurer_name, source_channel, currency, important_exclusions, "
                    " annual_total_premium, is_demonstration, created_at) "
                    "VALUES (?, ?, ?, ?, 'zurich', 'Zurich', 'direct', 'EUR', '[]', "
                    " '342.31', ?, ?)",
                    (quote_id, tenant, request, attempt, is_demo, now),
                )
            connection.commit()
        finally:
            connection.close()

        alembic("upgrade", "head", database_url=database)

        connection = sqlite3.connect(path)
        try:
            rows = dict(
                connection.execute(
                    "SELECT id, calculation_source FROM pc_normalized_quotes"
                ).fetchall()
            )
            premium = connection.execute(
                "SELECT annual_total_premium FROM pc_normalized_quotes WHERE id = ?",
                (demo_quote,),
            ).fetchone()[0]
        finally:
            connection.close()

        assert rows[demo_quote] == "demonstration_formula"
        assert rows[live_quote] == "provider_supplied"
        # The pre-existing data is untouched, and money is still an exact string.
        assert premium == "342.31"

    def test_no_schema_drift_against_the_models(self, database):
        """The migrations and the model definitions must agree."""
        script = (
            "from alembic.autogenerate import compare_metadata\n"
            "from alembic.migration import MigrationContext\n"
            "from policy_comparator.db import get_engine, Base\n"
            "from policy_comparator import models\n"
            "with get_engine().connect() as c:\n"
            "    diff = [d for d in compare_metadata(MigrationContext.configure(c), "
            "Base.metadata) if 'alembic_version' not in str(d)]\n"
            "print('DRIFT' if diff else 'CLEAN', diff)\n"
        )
        alembic("upgrade", "head", database_url=database)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PC_DATABASE_URL": database, "PC_MODE": "development"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("CLEAN"), result.stdout
