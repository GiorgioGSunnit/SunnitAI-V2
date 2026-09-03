"""Test support: an engine backed by a controlled parameter fixture.

The FOI calculation tests must not couple to the production ISTAT series —
that series changes as real months are ingested, and the production file only
carries the handful of months verified so far. These tests instead run
against this fixture: deliberately synthetic FOI months (clearly marked
placeholders) plus one cross-base pair, while reusing the real formula packs
and the real non-FOI parameter tables (legal interest rates, DM 55, ...).

Keeping the dev/test scenario data separate from production data is the whole
point: production stays honest (real values only), tests stay stable.
"""

import shutil
import tempfile
from pathlib import Path

from app.core.engine import CalculationEngine
from app.core.registry import CalculatorRegistry
from app.main import FORMULA_PACKS_DIR, PARAMETERS_DIR
from app.resolvers.parameter_store import ParameterStore

# The registry only reads the formula packs (no parameter data), so it is safe
# to build once and share across fixture engines.
_registry = CalculatorRegistry(FORMULA_PACKS_DIR)

# Synthetic FOI series for tests. The four base-2015 months reproduce the
# historical schema placeholders (so the golden hand-calculations that depend
# on them stay valid); the 2021-03 / 2026-06 pair is a cross-base scenario for
# the base-link relinking path.
FOI_FIXTURE = """\
base_links:
  - parameter_id: legal_it.foi_index
    from_base: 2025
    to_base: 2015
    coefficient: 1.214
    source: "ISTAT — coefficiente di raccordo base 2025 con base 2015 (fixture)"

values:
  - {parameter_id: legal_it.foi_index, year: 2024, month: 11, value: "100.0", base_year: 2015, unit: index, official: false, verified: false, placeholder: true, source_ref: "PLACEHOLDER fixture 2024-11"}
  - {parameter_id: legal_it.foi_index, year: 2024, month: 12, value: "100.5", base_year: 2015, unit: index, official: false, verified: false, placeholder: true, source_ref: "PLACEHOLDER fixture 2024-12"}
  - {parameter_id: legal_it.foi_index, year: 2025, month: 12, value: "102.0", base_year: 2015, unit: index, official: false, verified: false, placeholder: true, source_ref: "PLACEHOLDER fixture 2025-12"}
  - {parameter_id: legal_it.foi_index, year: 2026, month: 2, value: "102.5", base_year: 2015, unit: index, official: false, verified: false, placeholder: true, source_ref: "PLACEHOLDER fixture 2026-02"}
  - {parameter_id: legal_it.foi_index, year: 2021, month: 3, value: "103.3", base_year: 2015, unit: index, official: false, verified: false, placeholder: true, source_ref: "PLACEHOLDER fixture 2021-03 base 2015"}
  - {parameter_id: legal_it.foi_index, year: 2026, month: 6, value: "102.8", base_year: 2025, unit: index, official: false, verified: false, placeholder: true, source_ref: "PLACEHOLDER fixture 2026-06 base 2025"}
"""


def build_engine(foi_yaml: str = FOI_FIXTURE):
    """An engine whose parameter store is a copy of the production tables with
    the FOI series replaced by `foi_yaml`. Returns (engine, parameter_store)."""
    tmp = Path(tempfile.mkdtemp(prefix="calc-foi-fixture-"))
    params = tmp / "parameters"
    shutil.copytree(PARAMETERS_DIR, params)
    (params / "legal_it" / "foi_indices.yml").write_text(foi_yaml, encoding="utf-8")
    store = ParameterStore(params)
    return CalculationEngine(_registry, store), store


engine, parameter_store = build_engine()
