from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from db.base_class import Base

class Log(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, nullable=True)
    type = Column(Integer, nullable=False)  # 0: info, 1: warning, 2: error
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)