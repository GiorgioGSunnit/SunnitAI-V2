"""Formula Registry — Phase 5.

Loads all active Formula records from PostgreSQL at startup and holds them
in memory for the lifetime of the process. Also caches their vector embeddings
as numpy arrays so the semantic router can do fast in-memory cosine similarity
without hitting the DB on every query.

Usage:
    from src.calculator.registry import registry

    registry.load()          # call once at startup
    record = registry.get_by_slug("penale_contrattuale")
    embeddings = registry.get_all_embeddings()  # [(slug, np.ndarray), ...]
    registry.refresh()       # reload without restart (after seeding new formulas)

Design notes:
- This is a module-level singleton. A single process has one registry instance.
- load() is idempotent: calling it twice rebuilds the in-memory cache from scratch.
- The Formula ORM is NOT imported at module load time to avoid DB connection
  attempts during import. All DB access is deferred to load().
- pgvector returns the embedding column as a list of floats; np.array() handles
  both list and ndarray, so the cast is safe regardless of driver version.
"""

import logging
from typing import Optional

import numpy as np

from src.calculator.models import FormulaRecord, ParameterDefinition

logger = logging.getLogger(__name__)


class FormulaRegistry:
    """In-memory cache of formula records and their embeddings.

    Thread safety: load() and refresh() are not thread-safe. In the current
    architecture they are called at startup (single-threaded) before the
    application accepts requests, so this is acceptable for the POC.
    """

    def __init__(self) -> None:
        # slug → FormulaRecord
        self._formulas: dict[str, FormulaRecord] = {}
        # ordered list of (slug, embedding_vector) for cosine similarity scan
        self._embeddings: list[tuple[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all active formulas from the DB. Call once at startup.

        Rebuilds both the slug→record map and the embedding list from scratch.
        Any formula with is_active=False is excluded.
        Any formula whose embedding column is NULL is included in _formulas
        but excluded from _embeddings (will never be a router candidate).
        """
        # Deferred import keeps DB connection out of module-load time
        from src.db.base import SessionLocal
        from src.db.models import Formula

        db = SessionLocal()
        try:
            rows = db.query(Formula).filter(Formula.is_active == True).all()
        finally:
            db.close()

        formulas: dict[str, FormulaRecord] = {}
        embeddings: list[tuple[str, np.ndarray]] = []

        for row in rows:
            # Deserialize JSONB parameter_schema → list[ParameterDefinition]
            params = [ParameterDefinition(**p) for p in row.parameter_schema]

            record = FormulaRecord(
                id=row.id,
                slug=row.slug,
                name_it=row.name_it,
                description_it=row.description_it,
                category=row.category,
                expression_type=row.expression_type,
                expression=row.expression,
                plugin_name=row.plugin_name,
                parameter_schema=params,
                source_norm=row.source_norm,
                # similarity_score is ephemeral — not loaded from DB
            )
            formulas[row.slug] = record

            if row.embedding is not None:
                # pgvector returns list[float] or ndarray — np.array handles both
                embeddings.append((row.slug, np.array(row.embedding, dtype=np.float32)))

        # Atomic swap so readers always see a consistent snapshot
        self._formulas = formulas
        self._embeddings = embeddings

        logger.info(
            "FormulaRegistry loaded: %d formulas, %d with embeddings",
            len(self._formulas),
            len(self._embeddings),
        )

    def get_by_slug(self, slug: str) -> Optional[FormulaRecord]:
        """Return the FormulaRecord for this slug, or None if not found."""
        return self._formulas.get(slug)

    def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Return the full embedding cache as (slug, vector) pairs.

        Used by the router to iterate all embeddings for cosine similarity.
        Returns an empty list if load() has not been called yet or no formula
        has a non-null embedding.
        """
        return self._embeddings

    def all_slugs(self) -> list[str]:
        """Return the slugs of all loaded formulas (active only)."""
        return list(self._formulas.keys())

    def is_loaded(self) -> bool:
        """True if load() has been called and at least one formula was found."""
        return bool(self._formulas)

    def refresh(self) -> None:
        """Reload from DB without restarting the process.

        Call this after running scripts/seed_formulas.py or after adding
        a new formula to the database while the service is running.
        """
        logger.info("FormulaRegistry refresh requested — reloading from DB")
        self.load()


# ---------------------------------------------------------------------------
# Module-level singleton — imported directly by router, extractor, API
# ---------------------------------------------------------------------------

registry = FormulaRegistry()
