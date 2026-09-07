"""Loads CalculatorDefinition YAML files from formula_packs/ into memory.

Adding a new calculator means dropping a new YAML file under formula_packs/
and (if it needs a new calculation shape) a new strategy — this file never
needs to change for that.

Only calculators named in release_manifest.yml are served. Everything else
is withheld here rather than anywhere downstream — this is the single
chokepoint every entry point goes through: matching, the planner, discovery
listings, tool schemas and `/calculate` all reach a definition via `get()`
or `definitions()`. Setting SUNNIT_ENABLE_DRAFT_PACKS releases everything
for development. Withheld packs are still parsed and validated, so a broken
one breaks the build instead of rotting unnoticed.

Withholding computation is only half of it: `is_disclosable()` exists so the
routes that read STORED results can apply the same decision. A user handed a
stored criminal-sentencing range is harmed exactly as much as one who
computed it.
"""

from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import yaml
from pydantic import ValidationError as PydanticValidationError

from ..schemas.calculator_definition import CalculatorDefinition
from ..strategies import STRATEGIES
from .definition_validator import validate_definition
from .errors import CalculatorNotFoundError, DefinitionValidationError
from .release_policy import RELEASE_FLAG_ENV_VAR, load_released_ids, override_enabled_by_env


class CalculatorRegistry:
    def __init__(
        self,
        formula_packs_dir: Path,
        enable_drafts: Optional[bool] = None,
        released_ids: Optional[FrozenSet[str]] = None,
    ):
        self.enable_drafts = override_enabled_by_env() if enable_drafts is None else enable_drafts
        self._released_ids = load_released_ids() if released_ids is None else frozenset(released_ids)
        self._definitions: Dict[str, CalculatorDefinition] = {}
        # Withheld packs, kept only so `get()` can say why the id is
        # refused instead of reporting a bare unknown id.
        self._withheld: Dict[str, CalculatorDefinition] = {}
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
            if not self.is_disclosable(definition.id):
                self._withheld[definition.id] = definition
                continue
            self._definitions[definition.id] = definition

    def is_released(self, calculator_id: str) -> bool:
        """Manifest membership, independent of the override — so a caller
        running with the override on can still tell that a calculator has
        not passed human verification."""
        return calculator_id in self._released_ids

    def is_disclosable(self, calculator_id: str) -> bool:
        """Whether this calculator may be served OR its stored results
        shown. The single predicate every entry point shares."""
        return self.enable_drafts or self.is_released(calculator_id)

    def withheld_error(self, calculator_id: str) -> CalculatorNotFoundError:
        """The refusal for a calculator that exists but is not released.
        Same shape as the unknown-id error so no caller has to distinguish
        'withheld' from 'absent' to handle it."""
        return CalculatorNotFoundError(
            f"Il calcolatore {calculator_id!r} non e stato validato per l'uso "
            "e non e disponibile.",
            details={
                "calculator_id": calculator_id,
                "released": False,
                "enable_with": RELEASE_FLAG_ENV_VAR,
                "available": sorted(self._definitions),
            },
        )

    def get(self, calculator_id: str) -> CalculatorDefinition:
        if calculator_id in self._withheld:
            raise self.withheld_error(calculator_id)
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
