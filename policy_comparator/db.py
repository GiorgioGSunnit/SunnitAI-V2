"""Database engine, session handling and portable column types.

The parent platform's models bind directly to PostgreSQL dialect types
(``postgresql.UUID``, ``JSONB``), which cannot run on SQLite. This sub-project
needs to run from a clean checkout on a laptop with no database server, so it
defines its own declarative base over dialect-neutral ``TypeDecorator`` columns
that map to native PostgreSQL types in production and to portable ones on
SQLite.
"""

from __future__ import annotations

import json
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator

from sqlalchemy import CHAR, String, Text, create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator

from .config import get_settings
from .crypto import decrypt_text, encrypt_text


class Base(DeclarativeBase):
    """Declarative base for every policy_comparator table."""


# ---------------------------------------------------------------------------
# Portable column types
# ---------------------------------------------------------------------------


class GUID(TypeDecorator):
    """UUID column: native ``uuid`` on PostgreSQL, ``CHAR(36)`` elsewhere."""

    impl = CHAR(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGUUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        return value if dialect.name == "postgresql" else str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class JSONColumn(TypeDecorator):
    """JSON column: ``JSONB`` on PostgreSQL, serialized ``TEXT`` elsewhere."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value, default=str)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" or isinstance(value, (dict, list)):
            return value
        return json.loads(value)


class Money(TypeDecorator):
    """Exact monetary amount.

    Stored as a decimal string so the value survives a SQLite round-trip
    without ever touching binary floating point. Prices are compared and summed
    as :class:`~decimal.Decimal` throughout the application.
    """

    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, float):
            raise TypeError(
                "Refusing to store a float as money — pass a Decimal or a string"
            )
        try:
            return str(Decimal(str(value)))
        except InvalidOperation as exc:  # pragma: no cover - defensive
            raise ValueError(f"Not a monetary amount: {value!r}") from exc

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return Decimal(value)


class EncryptedString(TypeDecorator):
    """Application-level encryption for sensitive personal data at rest.

    Encryption happens in the application, not the database, so the ciphertext
    is identical on SQLite and PostgreSQL and a database dump never contains
    plaintext tax codes, addresses or phone numbers.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return encrypt_text(str(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return decrypt_text(value)


# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _create_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        # check_same_thread=False lets the FastAPI threadpool and the worker
        # share a file-backed SQLite database.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs.update(pool_size=5, max_overflow=10)
    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver glue
            cursor = dbapi_connection.cursor()
            # WAL keeps the worker's writes from blocking the API's reads.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_settings().database_url
        if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
            from pathlib import Path

            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        _engine = _create_engine(url)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autocommit=False, autoflush=False, expire_on_commit=False
        )
    return _session_factory


def session_scope() -> Session:
    """A new session. The caller owns commit/rollback/close."""
    return get_session_factory()()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = session_scope()
    try:
        yield db
    finally:
        db.close()


def reset_engine() -> None:
    """Dispose the engine and session factory. Used by tests."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def create_all() -> None:
    """Create every table. Local development and tests only.

    Production uses the Alembic migrations under ``migrations/``.
    """
    from . import models  # noqa: F401  (import registers the mappers)

    Base.metadata.create_all(bind=get_engine())
