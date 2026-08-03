"""Runtime configuration, read from the environment.

Everything that differs between a laptop, CI and production lives here. No
credential is ever stored in this file: provider secrets are referenced by the
*name* of the environment variable that holds them, and resolved lazily through
:mod:`policy_comparator.secrets_provider`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

Mode = Literal["development", "test", "production"]

#: Provider ids known to the application. Adapters register against these.
PROVIDER_IDS = ("zurich", "allianz", "generali", "cercassicurazioni")


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ProviderSettings:
    """Per-provider configuration.

    ``mode`` selects which implementation the adapter uses:

    ``mock``
        Deterministic demonstration data. Never touches the network.
    ``api``
        Official/partner API. Requires ``api_base_url`` and a credential ref.
    ``browser``
        Authorized portal automation via Playwright. Requires ``portal_url``.

    ``authorized`` records that the client has a contractual/authorized
    relationship with this provider. It is a hard precondition for any live
    submission — configuration alone is never enough.
    """

    provider_id: str
    mode: Literal["mock", "api", "browser"] = "mock"
    authorized: bool = False
    api_base_url: str | None = None
    portal_url: str | None = None
    #: Names of environment variables holding the secrets (never the secrets).
    api_key_env: str | None = None
    username_env: str | None = None
    password_env: str | None = None
    timeout_seconds: int | None = None
    retry_count: int | None = None

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"


def _provider_settings(provider_id: str) -> ProviderSettings:
    prefix = f"PC_PROVIDER_{provider_id.upper()}"
    raw_mode = (os.getenv(f"{prefix}_MODE") or "mock").strip().lower()
    if raw_mode not in {"mock", "api", "browser"}:
        raw_mode = "mock"
    return ProviderSettings(
        provider_id=provider_id,
        mode=raw_mode,  # type: ignore[arg-type]
        authorized=_bool(f"{prefix}_AUTHORIZED", False),
        api_base_url=os.getenv(f"{prefix}_API_BASE_URL") or None,
        portal_url=os.getenv(f"{prefix}_PORTAL_URL") or None,
        api_key_env=os.getenv(f"{prefix}_API_KEY_ENV") or None,
        username_env=os.getenv(f"{prefix}_USERNAME_ENV") or None,
        password_env=os.getenv(f"{prefix}_PASSWORD_ENV") or None,
        timeout_seconds=_int(f"{prefix}_TIMEOUT_SECONDS", 0) or None,
        retry_count=_int(f"{prefix}_RETRY_COUNT", -1) if os.getenv(f"{prefix}_RETRY_COUNT") else None,
    )


@dataclass(frozen=True)
class Settings:
    mode: Mode = "development"
    database_url: str = ""
    encryption_key: str | None = None
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720

    #: Master switch. Without it every adapter stays in mock mode regardless of
    #: per-provider configuration.
    live_provider_automation: bool = False

    worker_concurrency: int = 2
    max_concurrent_providers: int = 4
    provider_timeout_seconds: int = 60
    provider_retry_count: int = 2
    provider_retry_backoff_seconds: float = 2.0
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_seconds: int = 300

    quote_rate_limit_per_hour: int = 60
    data_retention_days: int = 365

    store_diagnostics: bool = True
    store_raw_pages: bool = False
    diagnostics_dir: Path = BASE_DIR / "var" / "diagnostics"

    providers: dict[str, ProviderSettings] = field(default_factory=dict)

    @property
    def is_production(self) -> bool:
        return self.mode == "production"

    def provider(self, provider_id: str) -> ProviderSettings:
        return self.providers.get(provider_id) or ProviderSettings(provider_id=provider_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw_mode = (os.getenv("PC_MODE") or "development").strip().lower()
    mode: Mode = raw_mode if raw_mode in {"development", "test", "production"} else "development"  # type: ignore[assignment]

    default_db = f"sqlite:///{BASE_DIR / 'var' / 'policy_comparator.db'}"
    database_url = os.getenv("PC_DATABASE_URL") or default_db

    diagnostics_dir = Path(
        os.getenv("PC_DIAGNOSTICS_DIR") or (BASE_DIR / "var" / "diagnostics")
    )

    return Settings(
        mode=mode,
        database_url=database_url,
        encryption_key=os.getenv("PC_ENCRYPTION_KEY") or None,
        # Shares the parent platform's signing secret so a token issued by the
        # main app is accepted here too. Standalone deployments just set it.
        jwt_secret_key=os.getenv("PC_JWT_SECRET_KEY")
        or os.getenv("JWT_SECRET_KEY")
        or "change-me-in-production",
        jwt_expire_minutes=_int("PC_JWT_EXPIRE_MINUTES", 720),
        live_provider_automation=_bool("LIVE_PROVIDER_AUTOMATION", False),
        worker_concurrency=_int("PC_WORKER_CONCURRENCY", 2),
        max_concurrent_providers=_int("PC_MAX_CONCURRENT_PROVIDERS", 4),
        provider_timeout_seconds=_int("PC_PROVIDER_TIMEOUT_SECONDS", 60),
        provider_retry_count=_int("PC_PROVIDER_RETRY_COUNT", 2),
        provider_retry_backoff_seconds=float(
            os.getenv("PC_PROVIDER_RETRY_BACKOFF_SECONDS") or 2.0
        ),
        circuit_breaker_threshold=_int("PC_CIRCUIT_BREAKER_THRESHOLD", 3),
        circuit_breaker_cooldown_seconds=_int("PC_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 300),
        quote_rate_limit_per_hour=_int("PC_QUOTE_RATE_LIMIT_PER_HOUR", 60),
        data_retention_days=_int("PC_DATA_RETENTION_DAYS", 365),
        store_diagnostics=_bool("PC_STORE_DIAGNOSTICS", True),
        store_raw_pages=_bool("PC_STORE_RAW_PAGES", False),
        diagnostics_dir=diagnostics_dir,
        providers={pid: _provider_settings(pid) for pid in PROVIDER_IDS},
    )


def reset_settings_cache() -> None:
    """Drop the memoized settings. Used by tests that patch the environment."""
    get_settings.cache_clear()
