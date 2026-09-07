from typing import Any, Dict, Optional, Union

from pydantic import BaseModel


class StoredCalculationSummary(BaseModel):
    request_id: str
    created_at: str
    calculator_id: str
    status: str
    result_preview: Optional[Union[Dict[str, Any], str]] = None


class StoredCalculation(StoredCalculationSummary):
    request: Dict[str, Any]
    result: Dict[str, Any]
