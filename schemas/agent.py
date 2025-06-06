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

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    system_instruction: Optional[str] = None
    agent_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    input_type: Optional[str] = None
    output_type: Optional[str] = None

class AgentInDB(AgentBase):
    id: int
    owner_id: int
    created_at: datetime
    is_enhanced: Optional[bool] = Field(default=False)  # Optional boolean, default False
    class Config:
        orm_mode = True