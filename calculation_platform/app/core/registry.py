"""Loads CalculatorDefinition YAML files from formula_packs/ into memory.

Adding a new calculator means dropping a new YAML file under formula_packs/
and (if it needs a new calculation shape) a new strategy — this file never
needs to change for that.

Draft packs (version ending "-draft") are legally unvalidated demonstrations
— criminal-sentencing ranges among them — so they are withheld here rather
than anywhere downstream. This is the single chokepoint every entry point
goes through: matching, the planner, discovery listings, tool schemas and
`/calculate` all reach a definition via `get()` or `definitions()`. Setting
SUNNIT_ENABLE_DRAFT_PACKS puts them back for development. They are still
parsed and validated when withheld, so a broken draft breaks the build
instead of rotting unnoticed.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError as PydanticValidationError

from ..schemas.calculator_definition import CalculatorDefinition
from ..strategies import STRATEGIES
from .definition_validator import validate_definition
from .errors import CalculatorNotFoundError, DefinitionValidationError

DRAFT_VERSION_SUFFIX = "-draft"
DRAFT_FLAG_ENV_VAR = "SUNNIT_ENABLE_DRAFT_PACKS"
_TRUTHY = {"1", "true", "yes", "on"}


def is_draft(definition: CalculatorDefinition) -> bool:
    return str(definition.version).endswith(DRAFT_VERSION_SUFFIX)


def drafts_enabled_by_env() -> bool:
    return os.environ.get(DRAFT_FLAG_ENV_VAR, "").strip().lower() in _TRUTHY


class CalculatorRegistry:
    def __init__(self, formula_packs_dir: Path, enable_drafts: Optional[bool] = None):
        self.enable_drafts = drafts_enabled_by_env() if enable_drafts is None else enable_drafts
        self._definitions: Dict[str, CalculatorDefinition] = {}
        # Withheld drafts, kept only so `get()` can say why the id is
        # refused instead of reporting a bare unknown id.
        self._withheld_drafts: Dict[str, CalculatorDefinition] = {}
        self._load(formula_packs_dir)

    def _load(self, formula_packs_dir: Path) -> None:
        for yml_path in sorted(formula_packs_dir.rglob("*.yml")):
            data = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
            try:
                definition = CalculatorDefinition(**data)
            except PydanticValidationError as e:
                raise DefinitionValidationError(
                    f"Malformed calculator definition in {yml_path}: {e}",
                    details={"file": str(yml_path)},
                ) from e
            validate_definition(definition, source_file=str(yml_path))
            definition.requires_period = STRATEGIES[definition.strategy].requires_period
            if is_draft(definition) and not self.enable_drafts:
                self._withheld_drafts[definition.id] = definition
                continue
            self._definitions[definition.id] = definition

    def get(self, calculator_id: str) -> CalculatorDefinition:
        if calculator_id in self._withheld_drafts:
            raise CalculatorNotFoundError(
                f"Il calcolatore {calculator_id!r} e una bozza non validata "
                "legalmente e non e disponibile.",
                details={
                    "calculator_id": calculator_id,
                    "draft": True,
                    "enable_with": DRAFT_FLAG_ENV_VAR,
                    "available": sorted(self._definitions),
                },
            )
        if calculator_id not in self._definitions:
            raise CalculatorNotFoundError(
                f"Unknown calculator_id: {calculator_id!r}",
                details={"calculator_id": calculator_id, "available": sorted(self._definitions)},
            )
        return self._definitions[calculator_id]

    def definitions(self) -> List[CalculatorDefinition]:
        return list(self._definitions.values())

    def list_all(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": d.id,
                "name": d.name,
                "category": d.category,
                "description": d.description,
                "keywords": d.keywords,
                "aliases": d.aliases,
            }
            for d in self._definitions.values()
        ]
