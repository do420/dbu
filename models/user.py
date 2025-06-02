from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from db.base_class import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
      # Relationships
    api_keys = relationship("APIKey", back_populates="owner")
    agents = relationship("Agent", back_populates="owner")
    mini_services = relationship("MiniService", back_populates="owner")
    processes = relationship("Process", back_populates="user")
    favorite_agents = relationship("FavoriteAgent", back_populates="user")