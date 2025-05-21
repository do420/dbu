from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from db.base_class import Base

class Process(Base):
    __tablename__ = "processes"
    id = Column(Integer, primary_key=True, index=True)
    mini_service_id = Column(Integer, ForeignKey("mini_services.id"), nullable=True)  # Made nullable for standalone agent operations
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    total_tokens = Column(JSON, default={}, nullable=False)  # Token usage details
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    mini_service = relationship("MiniService", back_populates="processes")
    user = relationship("User", back_populates="processes")