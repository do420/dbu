from db.base_class import Base

# Import all models
from models.user import User
from models.api_key import APIKey
from models.agent import Agent
from models.mini_service import MiniService
from models.process import Process
from models.log import Log
from models.favorite_service import FavoriteService
from models.favorite_agent import FavoriteAgent

# This ensures that all models are registered with SQLAlchemy metadata
__all__ = [
    "Base", 
    "User", 
    "APIKey", 
    "Agent", 
    "MiniService", 
    "Process", 
    "Log",
    "FavoriteService",
    "FavoriteAgent"
]