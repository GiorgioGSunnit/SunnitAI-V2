"""Test configuration for the policy comparator.

The environment is pinned *before* anything imports
:mod:`policy_comparator.config`, because settings are memoized on first read.
Two guarantees matter here:

* tests run against a throwaway SQLite file, never a developer's database;
* every provider stays in mock mode with zero simulated latency, so no test can
  reach a real insurer no matter how it is invoked.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="policy-comparator-tests-"))

os.environ.setdefault("PC_MODE", "test")
os.environ["PC_DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
# A fixed, obviously-not-secret Fernet key so encrypted columns round-trip
# identically on every run.
os.environ["PC_ENCRYPTION_KEY"] = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="
os.environ["PC_JWT_SECRET_KEY"] = "test-signing-secret"
os.environ["PC_DIAGNOSTICS_DIR"] = str(_TMP / "diagnostics")

# No provider may be contacted for real from a test run, whatever else is set
# in the developer's shell.
os.environ["LIVE_PROVIDER_AUTOMATION"] = "false"
os.environ["PC_MOCK_LATENCY_MS"] = "0"
for _provider in ("ZURICH", "ALLIANZ", "GENERALI", "CERCASSICURAZIONI"):
    os.environ[f"PC_PROVIDER_{_provider}_MODE"] = "mock"
    os.environ[f"PC_PROVIDER_{_provider}_AUTHORIZED"] = "false"
    os.environ.pop(f"PC_MOCK_FORCE_OUTCOME_{_provider}", None)
