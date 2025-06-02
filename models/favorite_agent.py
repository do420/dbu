from sqlalchemy import Column, Integer, DateTime, Index, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from db.base_class import Base

class FavoriteAgent(Base):
    __tablename__ = "favorite_agents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="favorite_agents")
    agent = relationship("Agent", back_populates="favorite_agents")
    
    # Create composite unique index to prevent duplicate favorites
    __table_args__ = (
        Index('idx_user_agent_unique', 'user_id', 'agent_id', unique=True),
    )
