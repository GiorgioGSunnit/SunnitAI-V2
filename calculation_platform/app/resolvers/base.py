from abc import ABC, abstractmethod
from typing import Any


class ParameterResolver(ABC):
    """Common interface for resolving a single named parameter to a value."""

    @abstractmethod
    def resolve(self, **context: Any) -> Any:
        ...
