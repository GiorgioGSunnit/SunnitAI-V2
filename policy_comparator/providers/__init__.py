"""Provider adapters. Import through :mod:`.registry`, never by module."""

from .base import ProviderAdapter, ProviderNotAuthorized
from .registry import (
    UnknownProvider,
    available_provider_ids,
    build_adapter,
    build_adapters,
    describe_providers,
)

__all__ = [
    "ProviderAdapter",
    "ProviderNotAuthorized",
    "UnknownProvider",
    "available_provider_ids",
    "build_adapter",
    "build_adapters",
    "describe_providers",
]
