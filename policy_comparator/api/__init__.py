"""HTTP layer. Thin translation between requests and the services package."""

from .app import create_app

__all__ = ["create_app"]
