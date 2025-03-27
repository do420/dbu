from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.session import get_db
from models.api_key import APIKey
from schemas.api_key import APIKeyCreate, APIKeyInDB
from core.security import encrypt_api_key, decrypt_api_key

router = APIRouter()

@router.post("/", response_model=APIKeyInDB)
async def create_api_key(
    api_key: APIKeyCreate, 
    db: Session = Depends(get_db),
    current_user_id: int = None  # Remove default value
):
    """Create a new API key"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Encrypt the API key before storing
    encrypted_key = encrypt_api_key(api_key.api_key)
    
    db_api_key = APIKey(
        api_key=encrypted_key,
        provider=api_key.provider,
        user_id=current_user_id
    )
    db.add(db_api_key)
    db.commit()
    db.refresh(db_api_key)
    
    return db_api_key

@router.get("/", response_model=List[APIKeyInDB])
async def list_api_keys(
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual auth
):
    """List all API keys for the current user"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    keys = db.query(APIKey).filter(APIKey.user_id == current_user_id).all()
    return keys

@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual auth
):
    """Delete an API key"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    db_key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == current_user_id
    ).first()
    
    if not db_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with ID {key_id} not found"
        )
    
    db.delete(db_key)
    db.commit()
    
    return None