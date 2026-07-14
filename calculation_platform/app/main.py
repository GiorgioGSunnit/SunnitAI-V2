from pathlib import Path

from fastapi import FastAPI

from .api.routes import router, set_engine
from .core.engine import CalculationEngine
from .core.registry import CalculatorRegistry
from .resolvers.parameter_store import ParameterStore
from .ui import ui_router, set_engine as set_ui_engine

BASE_DIR = Path(__file__).resolve().parent.parent
FORMULA_PACKS_DIR = BASE_DIR / "formula_packs"
PARAMETERS_DIR = BASE_DIR / "parameters"

registry = CalculatorRegistry(FORMULA_PACKS_DIR)
parameter_store = ParameterStore(PARAMETERS_DIR)
engine = CalculationEngine(registry, parameter_store)

app = FastAPI(title="Calculation Platform", version="0.1.0")
set_engine(engine)
set_ui_engine(engine)
app.include_router(router)
app.include_router(ui_router)
