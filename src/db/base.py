"""Database connection and session management."""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://astrea_admin:+Astrea01%25IT@localhost/astrea"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,    # checks connection is alive before using it
    pool_size=10,          # max connections in pool
    max_overflow=20        # extra connections allowed under load
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    """Dependency for FastAPI endpoints — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()