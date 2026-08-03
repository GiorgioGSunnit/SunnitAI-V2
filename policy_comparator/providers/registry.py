"""Provider registry.

Adding a provider means writing its module and adding one line here. Nothing
else in the application refers to a provider by name.
"""

from __future__ import annotations

from typing import Iterable, Type

from ..config import Settings, get_settings
from .allianz import AllianzAdapter
from .base import ProviderAdapter
from .cercassicurazioni import CercAssicurazioniAdapter
from .generali import GeneraliAdapter
from .zurich import ZurichAdapter

ADAPTER_CLASSES: tuple[Type[ProviderAdapter], ...] = (
    ZurichAdapter,
    AllianzAdapter,
    GeneraliAdapter,
    CercAssicurazioniAdapter,
)

_BY_ID: dict[str, Type[ProviderAdapter]] = {cls.provider_id: cls for cls in ADAPTER_CLASSES}


class UnknownProvider(KeyError):
    """Raised when a request names a provider that has no adapter."""


def available_provider_ids() -> tuple[str, ...]:
    return tuple(_BY_ID)


def adapter_class(provider_id: str) -> Type[ProviderAdapter]:
    try:
        return _BY_ID[provider_id]
    except KeyError as exc:
        raise UnknownProvider(provider_id) from exc


def build_adapter(provider_id: str, settings: Settings | None = None) -> ProviderAdapter:
    """Instantiate one adapter with its configuration.

    A fresh instance per attempt: adapters may hold per-attempt state (an HTTP
    client, a browser context) and must never be shared between customers.
    """
    settings = settings or get_settings()
    cls = adapter_class(provider_id)
    return cls(settings, settings.provider(provider_id))


def build_adapters(
    provider_ids: Iterable[str], settings: Settings | None = None
) -> dict[str, ProviderAdapter]:
    settings = settings or get_settings()
    return {pid: build_adapter(pid, settings) for pid in provider_ids}


def describe_providers(settings: Settings | None = None) -> list[dict]:
    """Static provider metadata for the provider-selection screen."""
    settings = settings or get_settings()
    out = []
    for cls in ADAPTER_CLASSES:
        config = settings.provider(cls.provider_id)
        out.append(
            {
                "provider_id": cls.provider_id,
                "display_name": cls.display_name,
                "provider_type": cls.provider_type.value,
                "reference_url": cls.reference_url,
                "mode": config.mode,
                "authorized": config.authorized,
                "live_enabled": (
                    settings.live_provider_automation
                    and config.authorized
                    and not config.is_mock
                ),
                "initial_fields": list(cls.required_paths),
            }
        )
    return out
