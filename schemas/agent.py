from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class AgentBase(BaseModel):
    name: str
    system_instruction: str
    agent_type: str
    config: Dict[str, Any]
    input_type: str
    output_type: str

class AgentCreate(AgentBase):
    pass

class AgentInDB(AgentBase):
    id: int
    owner_id: int
    created_at: datetime
    is_enhanced: Optional[bool] = Field(default=False)  # Optional boolean, default False
    class Config:
        orm_mode = True