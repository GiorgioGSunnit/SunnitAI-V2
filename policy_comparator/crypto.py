"""Symmetric encryption for sensitive stored fields.

Production must supply ``PC_ENCRYPTION_KEY`` (a urlsafe base64 Fernet key,
generate one with ``python -m policy_comparator.cli genkey``). There is
deliberately **no** production fallback: starting the app in production without
a key raises rather than silently encrypting everything under a known constant.

Development and test runs without a key derive a clearly-labelled local key so
a fresh checkout works with zero setup. That key is not a secret and is never
used when ``PC_MODE=production``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from .config import get_settings

_PREFIX = "pcenc1:"

_DEV_KEY_MATERIAL = b"policy_comparator-insecure-development-key-do-not-use-in-production"


class EncryptionNotConfigured(RuntimeError):
    """Raised when production is missing ``PC_ENCRYPTION_KEY``."""


def generate_key() -> str:
    """A fresh Fernet key, suitable for ``PC_ENCRYPTION_KEY``."""
    return Fernet.generate_key().decode()


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    settings = get_settings()
    raw = settings.encryption_key
    if raw:
        try:
            return Fernet(raw.encode() if isinstance(raw, str) else raw)
        except (ValueError, TypeError) as exc:
            raise EncryptionNotConfigured(
                "PC_ENCRYPTION_KEY is not a valid Fernet key. Generate one with "
                "`python -m policy_comparator.cli genkey`."
            ) from exc
    if settings.is_production:
        raise EncryptionNotConfigured(
            "PC_ENCRYPTION_KEY must be set when PC_MODE=production. Generate one "
            "with `python -m policy_comparator.cli genkey`."
        )
    derived = base64.urlsafe_b64encode(hashlib.sha256(_DEV_KEY_MATERIAL).digest())
    return Fernet(derived)


def reset_crypto_cache() -> None:
    """Drop the memoized cipher. Used by tests that patch the environment."""
    _fernet.cache_clear()


def blind_index(value: str) -> str:
    """A deterministic, non-reversible lookup token for an encrypted column.

    Fernet ciphertext is randomized, so an encrypted email cannot be searched
    for directly. This derives a stable HMAC from the same key material, which
    allows equality lookups without storing the value in clear.
    """
    key = _fernet_key_material()
    digest = hmac.new(key, value.strip().lower().encode(), hashlib.sha256)
    return digest.hexdigest()


def _fernet_key_material() -> bytes:
    settings = get_settings()
    raw = settings.encryption_key
    if raw:
        return raw.encode() if isinstance(raw, str) else raw
    if settings.is_production:
        raise EncryptionNotConfigured(
            "PC_ENCRYPTION_KEY must be set when PC_MODE=production."
        )
    return base64.urlsafe_b64encode(hashlib.sha256(_DEV_KEY_MATERIAL).digest())


def encrypt_text(value: str) -> str:
    return _PREFIX + _fernet().encrypt(value.encode()).decode()


def decrypt_text(value: str) -> str:
    """Decrypt a stored value.

    Values written before encryption was enabled (no prefix) are passed
    through unchanged so an existing database stays readable.
    """
    if not value.startswith(_PREFIX):
        return value
    token = value[len(_PREFIX) :]
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise EncryptionNotConfigured(
            "Stored value could not be decrypted with the configured "
            "PC_ENCRYPTION_KEY — the key has changed or the data belongs to "
            "another environment."
        ) from exc
