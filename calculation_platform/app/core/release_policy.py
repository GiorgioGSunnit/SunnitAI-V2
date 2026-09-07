"""Which calculators are cleared for release, and the override that lifts it.

The release decision is an allowlist read from release_manifest.yml, kept
deliberately outside formula_packs/: a pack author must not be able to
publish a legally unvalidated calculator by editing the same file they
wrote the formula in.

Default-deny is the whole point. The previous control keyed on a version
string ending "-draft", which failed open for every authoring mistake it
existed to contain — '0.1-DRAFT', '1.0-draft.1', '1.0-draft+build', a
trailing space, and above all a brand-new pack whose author simply wrote
version "1.0". An allowlist has no such failure mode: not being on the
list is the default state of everything.
"""

import os
from pathlib import Path
from typing import Any, FrozenSet, Optional

import yaml

from .errors import DefinitionValidationError

RELEASE_FLAG_ENV_VAR = "SUNNIT_ENABLE_DRAFT_PACKS"
_TRUTHY = {"1", "true", "yes", "on"}

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "release_manifest.yml"


def override_enabled_by_env() -> bool:
    """True when the environment releases everything (development/tests)."""
    return os.environ.get(RELEASE_FLAG_ENV_VAR, "").strip().lower() in _TRUTHY


def load_released_ids(manifest_path: Optional[Path] = None) -> FrozenSet[str]:
    """Read the set of released calculator ids.

    A missing or malformed manifest is a load-time failure, not an empty
    allowlist: silently releasing nothing would take the whole platform
    down, and silently releasing everything would be far worse.
    """
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise DefinitionValidationError(
            f"Release manifest not found at {path}; every calculator would be withheld.",
            details={"manifest": str(path)},
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or "released" not in data:
        raise DefinitionValidationError(
            f"Release manifest {path} must be a mapping with a 'released' key.",
            details={"manifest": str(path)},
        )

    entries = data.get("released") or []
    if not isinstance(entries, list):
        raise DefinitionValidationError(
            f"Release manifest {path}: 'released' must be a list.",
            details={"manifest": str(path)},
        )

    released = set()
    for entry in entries:
        calculator_id = _entry_id(entry, path)
        if calculator_id in released:
            raise DefinitionValidationError(
                f"Release manifest {path}: duplicate entry {calculator_id!r}.",
                details={"manifest": str(path), "calculator_id": calculator_id},
            )
        released.add(calculator_id)
    return frozenset(released)


def _entry_id(entry: Any, path: Path) -> str:
    """A manifest entry is a bare id, or a mapping carrying `id` plus
    free-form annotations (verified_on, verified_by, notes)."""
    if isinstance(entry, str):
        candidate = entry.strip()
    elif isinstance(entry, dict):
        candidate = str(entry.get("id", "")).strip()
    else:
        candidate = ""
    if not candidate:
        raise DefinitionValidationError(
            f"Release manifest {path}: entry {entry!r} declares no calculator id.",
            details={"manifest": str(path), "entry": entry},
        )
    return candidate
