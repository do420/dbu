from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from db.session import get_db
from models.favorite_service import FavoriteService
from models.mini_service import MiniService
from schemas.favorite_service import (
    FavoriteServiceCreate, 
    FavoriteServiceInDB, 
    FavoriteCountResponse
)
from core.log_utils import create_log
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=FavoriteServiceInDB)
async def add_favorite_service(
    favorite: FavoriteServiceCreate,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Add a mini service to user's favorites"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Check if mini service exists
    mini_service = db.query(MiniService).filter(
        MiniService.id == favorite.mini_service_id
    ).first()
    
    if not mini_service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mini service with ID {favorite.mini_service_id} not found"
        )
    
    # Check if already favorited
    existing_favorite = db.query(FavoriteService).filter(
        FavoriteService.user_id == current_user_id,
        FavoriteService.mini_service_id == favorite.mini_service_id
    ).first()
    
    if existing_favorite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Mini service is already in your favorites"
        )
    
    # Create favorite
    db_favorite = FavoriteService(
        user_id=current_user_id,
        mini_service_id=favorite.mini_service_id
    )
    
    db.add(db_favorite)
    
    # Log the action
    create_log(
        db=db,
        user_id=current_user_id,
        log_type=0,  # info
        description=f"Added mini-service '{mini_service.name}' to favorites"
    )
    
    db.commit()
    db.refresh(db_favorite)
    
    return db_favorite


@router.delete("/{mini_service_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_favorite_service(
    mini_service_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Remove a mini service from user's favorites"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Find the favorite
    favorite = db.query(FavoriteService).filter(
        FavoriteService.user_id == current_user_id,
        FavoriteService.mini_service_id == mini_service_id
    ).first()
    
    if not favorite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Favorite not found"
        )
    
    # Get mini service name for logging
    mini_service = db.query(MiniService).filter(
        MiniService.id == mini_service_id
    ).first()
    
    # Delete favorite
    db.delete(favorite)
    
    # Log the action
    if mini_service:
        create_log(
            db=db,
            user_id=current_user_id,
            log_type=0,  # info
            description=f"Removed mini-service '{mini_service.name}' from favorites"
        )
    
    db.commit()


@router.get("/", response_model=List[FavoriteServiceInDB])
async def list_user_favorites(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """List user's favorite mini services"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    favorites = db.query(FavoriteService).filter(
        FavoriteService.user_id == current_user_id
    ).offset(skip).limit(limit).all()
    
    return favorites


@router.get("/count/{mini_service_id}", response_model=FavoriteCountResponse)
async def get_favorite_count(
    mini_service_id: int,
    db: Session = Depends(get_db)
):
    """Get the favorite count for a specific mini service"""
    # Check if mini service exists
    mini_service = db.query(MiniService).filter(
        MiniService.id == mini_service_id
    ).first()
    
    if not mini_service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mini service with ID {mini_service_id} not found"
        )
    
    # Count favorites
    favorite_count = db.query(func.count(FavoriteService.id)).filter(
        FavoriteService.mini_service_id == mini_service_id
    ).scalar()
    
    return FavoriteCountResponse(
        mini_service_id=mini_service_id,
        favorite_count=favorite_count
    )


@router.get("/check/{mini_service_id}")
async def check_if_favorited(
    mini_service_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Check if a mini service is in user's favorites"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    favorite = db.query(FavoriteService).filter(
        FavoriteService.user_id == current_user_id,
        FavoriteService.mini_service_id == mini_service_id
    ).first()
    
    return {"is_favorited": favorite is not None}
