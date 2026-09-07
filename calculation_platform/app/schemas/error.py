from typing import Any, Dict

from pydantic import BaseModel, Field


class CalculationError(BaseModel):
    """A single structured, machine-readable failure — replaces plain
    error strings so a caller can branch on `code` instead of parsing
    `message`."""

    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
