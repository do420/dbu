# You should have this in core/security.py
from typing import Optional
from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from core.config import settings
from cryptography.fernet import Fernet
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API key encryption
# Generate a key for Fernet encryption
#FERNET_KEY = os.getenv("SECRET_KEY")
#random key:
FERNET_KEY = "W7KtUnP-4-ShzOpaJ_BVEeojck4igjAmOmPZ1PWtYSE="
cipher_suite = Fernet(FERNET_KEY)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm="HS256"
    )
    return encoded_jwt


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key before storing in database"""
    return cipher_suite.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key from database"""
    return cipher_suite.decrypt(encrypted_key.encode()).decode()