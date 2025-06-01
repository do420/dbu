from typing import List, Optional
from pydantic import BaseModel


class VoiceResponse(BaseModel):
    """Schema for TTS voice response"""
    Name: str
    ShortName: str
    Gender: str
    Locale: str
    SuggestedCodec: str
    FriendlyName: str
    Status: str
    VoiceTag: str  # Convert complex VoiceTag to simple string
    
    class Config:
        # Allow extra fields but don't include them in serialization
        extra = "ignore"


class SimpleVoiceResponse(BaseModel):
    """Simplified voice response for easier consumption"""
    name: str
    short_name: str
    gender: str
    locale: str
    friendly_name: str
    status: str