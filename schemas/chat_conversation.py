from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class ChatConversationBase(BaseModel):
    mini_service_id: int
    conversation: List[Dict[str, Any]] = []

class ChatConversationCreate(ChatConversationBase):
    pass

class ChatConversationUpdate(BaseModel):
    conversation: List[Dict[str, Any]]

class ChatConversationInDB(ChatConversationBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None
