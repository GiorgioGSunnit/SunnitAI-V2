"""Audit trail and PII-safe logging.

Two rules the rest of the application depends on:

* audit metadata records field *names*, never field *values*;
* application logs never carry personal data — identifiers are UUIDs, and
  anything free-form is scrubbed before it is written.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from sqlalchemy.orm import Session

from ..models import AuditEvent
from ..models.enums import AuditAction

logger = logging.getLogger("policy_comparator")

#: Patterns scrubbed from anything heading for a log line.
_SCRUBBERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b", re.I), "[TAX_CODE]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+39\s?)?3\d{2}[\s.-]?\d{6,7}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b[A-Z]{2}\s?\d{3}\s?[A-Z]{2}\b"), "[PLATE]"),
)


def scrub(text: str | None) -> str | None:
    """Remove the personal data that most often leaks into free-form strings."""
    if not text:
        return text
    for pattern, replacement in _SCRUBBERS:
        text = pattern.sub(replacement, text)
    return text


def log_event(level: int, message: str, **context: Any) -> None:
    """Structured log line with every value scrubbed."""
    safe = {k: scrub(v) if isinstance(v, str) else v for k, v in context.items()}
    logger.log(level, message, extra={"pc": safe})


def record(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    action: AuditAction,
    actor_user_id: uuid.UUID | None = None,
    actor_email: str | None = None,
    entity_type: str | None = None,
    entity_id: uuid.UUID | None = None,
    provider_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    """Append one audit event.

    ``metadata`` is expected to be structural — counts, field names, status
    codes. Callers must not put customer values in it.
    """
    event = AuditEvent(
        tenant_id=tenant_id,
        actor_user_id=actor_user_id,
        actor_email=actor_email,
        action=action.value,
        entity_type=entity_type,
        entity_id=entity_id,
        provider_id=provider_id,
        metadata_json=metadata or {},
    )
    db.add(event)
    db.flush()
    return event
