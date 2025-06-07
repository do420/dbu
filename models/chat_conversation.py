from sqlalchemy import Column, Integer, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship 
from datetime import datetime
from db.base_class import Base

class ChatConversation(Base):
    __tablename__ = "chat_conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    mini_service_id = Column(Integer, ForeignKey("mini_services.id"), nullable=False)
    conversation = Column(JSON, default=[], nullable=False)  # Store conversation history as JSON array
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="chat_conversations")
    mini_service = relationship("MiniService", back_populates="chat_conversations")
