from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import Depends
from sqlalchemy.orm import Session
from core.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.base_class import Base  # Import Base from base_class instead

SQLALCHEMY_DATABASE_URL = settings.SQLALCHEMY_DATABASE_URL

# Create the engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# Create the sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()