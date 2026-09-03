import os
import sys
import tempfile
from pathlib import Path

# Makes `app` importable as a top-level package regardless of where pytest
# is invoked from, without needing calculation_platform installed as a
# package or added to the main project's src/ layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Prevent app.main import-time setup from ever targeting calculation_platform/data
# during pytest collection, regardless of which test module imports it first.
os.environ["CALC_DB_PATH"] = str(
    Path(tempfile.mkdtemp(prefix="calc-api-import-")) / "calculations.db"
)
