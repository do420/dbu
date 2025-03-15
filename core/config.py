from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Superapp"
    SQLALCHEMY_DATABASE_URL: str = "sqlite:///./test.db"
    SECRET_KEY: str = "supersecretkey"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()
# In your core/config.py
SECRET_KEY = "your-secret-key-here"  # Change this to a secure random string
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # or whatever duration you prefer