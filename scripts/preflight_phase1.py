from pgvector.sqlalchemy import Vector # noqa: F401
from src.db.base import engine
from sqlalchemy import text
import alembic
"""
Phase 1 Pre-flight Checks — Formula Engine Data Layer
======================================================
Run this BEFORE executing the Alembic migration.
Usage: python scripts/preflight_phase1.py

Checks:
  1. pgvector Python package is importable
  2. pgvector extension is available on the PostgreSQL instance
  3. alembic is available and the migration file is present
"""
import sys
#import os
from pathlib import Path

# ── Make sure src is importable from project root ─────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"

errors = []


def check(label: str, ok: bool, detail: str = ""):
    symbol = PASS if ok else FAIL
    print(f"  {symbol}  {label}" + (f"  →  {detail}" if detail else ""))
    if not ok:
        errors.append(label)


print("\n=== Phase 1 Pre-flight Checks ===\n")

# ── Check 1: pgvector Python package ─────────────────────────────────────────
try:

    check("pgvector Python package", True, "importable")
except ImportError as e:
    check("pgvector Python package", False,
          f"NOT installed — run: pip install pgvector>=0.2.5  ({e})")

# ── Check 2: DB connection + pgvector extension ───────────────────────────────
try:

    with engine.connect() as conn:
        # Check extension available
        row = conn.execute(
            text("SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector'")
        ).fetchone()
        if row:
            check("pgvector extension available in PostgreSQL", True,
                  f"version {row[1]}")
        else:
            check("pgvector extension available in PostgreSQL", False,
                  "NOT available — install postgresql-<ver>-pgvector at OS level")

        # Check if already installed — informational only, migration handles this
        installed = conn.execute(
            text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()
        if installed:
            print(f"  {PASS}  pgvector extension already installed in DB  →  version {installed[1]}")
        else:
            # NOT a hard failure — migration runs CREATE EXTENSION IF NOT EXISTS vector
            print(f"  {WARN}  pgvector extension not yet installed in DB  "
                  "→  migration will install it automatically (this is expected on a fresh DB)")

except Exception as e:
    check("PostgreSQL connection", False, str(e))

# ── Check 3: alembic available ────────────────────────────────────────────────
try:

    check("alembic importable", True, f"version {alembic.__version__}")
except ImportError as e:
    check("alembic importable", False, str(e))

# ── Check 4: migration file exists ───────────────────────────────────────────
migration_path = ROOT / "src" / "db" / "migrations" / "versions" / "0002_add_formula_registry.py"
check("migration file 0002 exists", migration_path.exists(), str(migration_path))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if not errors:
    print("All checks passed — safe to run: alembic upgrade head\n")
else:
    print(f"FAILED checks ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
    print("\nResolve the above before running the migration.\n")
    sys.exit(1)
