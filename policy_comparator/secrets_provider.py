"""Indirection between provider configuration and the actual secret values.

Configuration stores the *name* of an environment variable, never its value, so
a leaked settings dump or an audit log can never contain a credential. Swapping
in a real secret manager (Vault, AWS Secrets Manager, ...) means replacing
:func:`resolve_secret` only.
"""

from __future__ import annotations

import os


class SecretNotConfigured(RuntimeError):
    """Raised when a configured secret reference resolves to nothing."""


def resolve_secret(ref: str | None) -> str | None:
    """Resolve a secret reference to its value, or ``None`` if unset.

    ``ref`` is the *name* of an environment variable, e.g. ``ZURICH_API_KEY``.
    """
    if not ref:
        return None
    value = os.getenv(ref)
    if value is None or not value.strip():
        return None
    return value


def require_secret(ref: str | None, *, purpose: str) -> str:
    value = resolve_secret(ref)
    if value is None:
        raise SecretNotConfigured(
            f"No secret available for {purpose}"
            + (f" (expected environment variable {ref})" if ref else " (no reference configured)")
        )
    return value
