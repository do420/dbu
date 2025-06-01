from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class FavoriteServiceBase(BaseModel):
    mini_service_id: int

class FavoriteServiceCreate(FavoriteServiceBase):
    pass

class FavoriteServiceInDB(FavoriteServiceBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        orm_mode = True

class FavoriteCountResponse(BaseModel):
    mini_service_id: int
    favorite_count: int
