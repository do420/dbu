from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class FavoriteAgentBase(BaseModel):
    agent_id: int

class FavoriteAgentCreate(FavoriteAgentBase):
    pass

class FavoriteAgentInDB(FavoriteAgentBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        orm_mode = True

class FavoriteAgentCountResponse(BaseModel):
    agent_id: int
    favorite_count: int
