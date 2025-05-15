from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class MiniServiceBase(BaseModel):
    name: str
    description: Optional[str] = None
    workflow: Dict[str, Any]
    input_type: str
    output_type: str

class MiniServiceCreate(MiniServiceBase):
    pass

class MiniServiceInDB(MiniServiceBase):
    id: int
    owner_id: int
    created_at: datetime
    average_token_usage: Dict[str, Any]
    run_time: int
    is_enhanced: Optional[bool] = False  # Changed to Optional with default=False

    class Config:
        orm_mode = True