from sqlalchemy import Column, Integer, DateTime, Index
from datetime import datetime
from db.base_class import Base

class FavoriteService(Base):
    __tablename__ = "favorite_services"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    mini_service_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Create composite unique index to prevent duplicate favorites
    __table_args__ = (
        Index('idx_user_service_unique', 'user_id', 'mini_service_id', unique=True),
    )
