from sqlalchemy import Boolean, Column, Integer, String, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from db.base_class import Base

class MiniService(Base):
    __tablename__ = "mini_services"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(String, nullable=True)
    workflow = Column(JSON, nullable=False)  # Start node and connected nodes
    input_type = Column(String, nullable=False)  # "text", "image", "sound"
    output_type = Column(String, nullable=False)  # "text", "image", "sound"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    average_token_usage = Column(JSON, default={}, nullable=False)
    run_time = Column(Integer, default=0, nullable=False)
    is_enhanced = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)  # New field for public/private visibility
    
    # Relationships
    owner = relationship("User", back_populates="mini_services")
    processes = relationship("Process", back_populates="mini_service")