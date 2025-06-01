from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class ProcessBase(BaseModel):
    mini_service_id: Optional[int] = None
    total_tokens: Dict[str, Any]

class ProcessCreate(ProcessBase):
    pass

class ProcessInDB(ProcessBase):
    id: int
    user_id: int
    created_at: datetime
    # Mini service information
    mini_service_name: Optional[str] = None
    mini_service_description: Optional[str] = None
    mini_service_input_type: Optional[str] = None
    mini_service_output_type: Optional[str] = None
    mini_service_is_enhanced: Optional[bool] = None

    class Config:
        orm_mode = True