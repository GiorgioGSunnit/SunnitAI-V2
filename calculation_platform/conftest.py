import sys
from pathlib import Path

# Makes `app` importable as a top-level package regardless of where pytest
# is invoked from, without needing calculation_platform installed as a
# package or added to the main project's src/ layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
