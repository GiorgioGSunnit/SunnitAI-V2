"""Helpers for building a consistent, auditable calculation trace.

Every strategy appends to the same kind of step list — a numbered entry
tagged with a `type` plus whatever strategy-specific detail fields make
sense for that step — so a caller can walk `steps` uniformly across every
calculator without knowing which strategy produced them, while each
strategy still records whatever fields are actually meaningful for its
own kind of step (e.g. a bracket step vs. an interest-segment step).
"""

from typing import Any, Dict, List


class AuditTrail:
    def __init__(self) -> None:
        self._steps: List[Dict[str, Any]] = []

    def record(self, step_type: str, **fields: Any) -> None:
        self._steps.append({"step": len(self._steps) + 1, "type": step_type, **fields})

    @property
    def steps(self) -> List[Dict[str, Any]]:
        return self._steps
