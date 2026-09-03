from decimal import Decimal
from typing import Any


def resolve_static(default: Any) -> Any:
    """Resolve a parameter that has no date-dependence — just a fixed
    default declared directly in the calculator definition's YAML."""
    if isinstance(default, (list, dict, bool, str)):
        return default
    return Decimal(str(default))
