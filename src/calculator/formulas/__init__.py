"""Plugin registry for complex formula implementations.

Simple formulas (expression_type='simple') are handled by SafeExpressionEvaluator
in executor.py. Complex formulas (expression_type='complex') require a dedicated
Python function registered here via the @register decorator.

Usage — registering a plugin:
    @register("my_formula_slug")
    def calculate_my_formula(params: dict) -> FormulaResult:
        ...

Usage — looking up a plugin (done by executor._run_plugin):
    plugin = get_plugin("my_formula_slug")   # returns None if not registered
"""

from typing import Callable

_PLUGINS: dict[str, Callable] = {}


def register(slug: str):
    """Decorator that registers a formula plugin function by slug.

    The decorated function receives a params dict and must return FormulaResult.
    """
    def decorator(fn: Callable) -> Callable:
        _PLUGINS[slug] = fn
        return fn
    return decorator


def get_plugin(slug: str) -> Callable | None:
    """Return the registered plugin for slug, or None if not found."""
    return _PLUGINS.get(slug)


# ---------------------------------------------------------------------------
# Auto-import all plugin modules so they self-register via @register.
# This MUST come after the registry functions above are defined — importing
# a plugin module triggers its @register call, which needs register() to exist.
# Add a new import here whenever a new complex formula plugin is added.
# ---------------------------------------------------------------------------
from src.calculator.formulas import tfr as _tfr_plugin  # noqa: F401, E402
