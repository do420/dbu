from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from db.base_class import Base

class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    system_instruction = Column(String, nullable=False)
    agent_type = Column(String, nullable=False)  # "gemini", "openai", "text2image", "text2speech"
    config = Column(JSON, nullable=False)  # Store temperature, model name etc.
    input_type = Column(String, nullable=False)  # "text", "image", "sound"
    output_type = Column(String, nullable=False)  # "text", "image", "sound"
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    owner = relationship("User", back_populates="agents")