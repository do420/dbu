from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class LogInDB(BaseModel):
    id: int
    description: str
    created_at: datetime
    user_id: int


    class Config:
        orm_mode = True
