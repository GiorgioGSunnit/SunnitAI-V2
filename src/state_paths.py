"""v77: the runtime state -- what the server writes or holds that is not code or shipped data: the sessions file, the
tenants' logos, the system templates, the article cards, the logs. It lives in one folder, outside src/ and data/, so
scripts/promote.sh can sync the code by directory without touching it and snapshot it whole:

  STATE_DIR (environment), else /opt/chatbot/state when /opt/chatbot exists (the live copy, and the test copy
  /root/v38, which has always read the live sessions, logos and templates), else <checkout>/state (a laptop, the tests).

  sessions.json  tenant_logos/  system_templates/  article_cards.jsonl  logs/

Each path keeps its own override (SESSIONS_FILE, SYSTEM_TEMPLATES_CATALOG_PATH, ARTICLE_CARDS_FILE, CYPHER_LOG_PATH,
THESIS_LOG_DIR). scripts/promote.sh moves the old locations (data/..., logs/) here once, with the service stopped."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = "STATE_DIR"
SERVER_DIR = Path("/opt/chatbot/state")


def state_dir() -> str:
    configured = (os.getenv(ENV) or "").strip()
    if configured:
        return configured
    return str(SERVER_DIR if SERVER_DIR.parent.is_dir() else ROOT / "state")


def path(*parts: str) -> str:
    return os.path.join(state_dir(), *parts)


def sessions_file() -> str:
    return path("sessions.json")


def tenant_logos_dir() -> str:
    return path("tenant_logos")


def system_templates_dir() -> str:
    return path("system_templates")


def system_templates_catalog() -> str:
    return path("system_templates", "catalog_enriched.json")


def article_cards_file() -> str:
    return path("article_cards.jsonl")


def logs_dir() -> str:
    return path("logs")
