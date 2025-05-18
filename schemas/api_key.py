from pydantic import BaseModel

class APIKeyBase(BaseModel):
    provider: str

class APIKeyCreate(APIKeyBase):
    api_key: str

class APIKeyInDB(APIKeyBase):
    id: int
    user_id: int
    api_key: str

    class Config:
        orm_mode = True