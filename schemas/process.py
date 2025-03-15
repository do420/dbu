from typing import Dict, Any
from pydantic import BaseModel
from datetime import datetime

class ProcessBase(BaseModel):
    mini_service_id: int
    total_tokens: Dict[str, Any]

class ProcessCreate(ProcessBase):
    pass

class ProcessInDB(ProcessBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        orm_mode = True