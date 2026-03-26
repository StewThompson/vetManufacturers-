from pydantic import BaseModel
from typing import Optional

class Manufacturer(BaseModel):
    name: str
    location: Optional[str] = None
    address: Optional[str] = None
    domain: Optional[str] = None
