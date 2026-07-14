from pydantic import BaseModel


class Warning(BaseModel):
    code: str
    message: str
