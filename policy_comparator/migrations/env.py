"""Alembic environment for the policy comparator.

The URL comes from ``PC_DATABASE_URL`` via the application settings, so
migrations always target the same database the application does.

``render_as_batch`` is on because SQLite cannot ALTER a column in place;
without it, any future column change would work on PostgreSQL and fail on a
developer's laptop.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from policy_comparator import models  # noqa: F401  (registers the mappers)
from policy_comparator.config import get_settings
from policy_comparator.db import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
