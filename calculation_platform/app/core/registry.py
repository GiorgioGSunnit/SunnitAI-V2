"""Loads CalculatorDefinition YAML files from formula_packs/ into memory.

Adding a new calculator means dropping a new YAML file under formula_packs/
and (if it needs a new calculation shape) a new strategy — this file never
needs to change for that.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import ValidationError as PydanticValidationError

from ..schemas.calculator_definition import CalculatorDefinition
from ..strategies import STRATEGIES
from .definition_validator import validate_definition
from .errors import CalculatorNotFoundError, DefinitionValidationError


class CalculatorRegistry:
    def __init__(self, formula_packs_dir: Path):
        self._definitions: Dict[str, CalculatorDefinition] = {}
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
            self._definitions[definition.id] = definition

    def get(self, calculator_id: str) -> CalculatorDefinition:
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
