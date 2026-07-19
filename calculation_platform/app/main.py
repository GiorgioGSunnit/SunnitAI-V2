import os
from pathlib import Path

from fastapi import FastAPI

from .api.routes import router, set_engine, set_store
from .core.engine import CalculationEngine
from .core.registry import CalculatorRegistry
from .resolvers.parameter_store import ParameterStore
from .storage.sqlite_store import SqliteCalculationStore
from .ui import ui_router, set_engine as set_ui_engine

BASE_DIR = Path(__file__).resolve().parent.parent
FORMULA_PACKS_DIR = BASE_DIR / "formula_packs"
PARAMETERS_DIR = BASE_DIR / "parameters"
DEFAULT_DB_PATH = BASE_DIR / "data" / "calculations.db"


def _db_path() -> Path:
    configured = os.environ.get("CALC_DB_PATH")
    return Path(configured) if configured else DEFAULT_DB_PATH

registry = CalculatorRegistry(FORMULA_PACKS_DIR)
parameter_store = ParameterStore(PARAMETERS_DIR)
engine = CalculationEngine(registry, parameter_store)
store = SqliteCalculationStore(_db_path())

app = FastAPI(title="Calculation Platform", version="0.1.0")
set_engine(engine)
set_store(store)
set_ui_engine(engine)
app.include_router(router)
app.include_router(ui_router)
