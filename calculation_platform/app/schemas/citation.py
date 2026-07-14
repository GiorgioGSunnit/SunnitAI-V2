from typing import Optional

from pydantic import BaseModel


class Citation(BaseModel):
    reference: str
    source_name: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    url: Optional[str] = None
    official: bool = False
