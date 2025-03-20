from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.session import get_db
from models.process import Process
from schemas.process import ProcessInDB

router = APIRouter()

@router.get("/", response_model=List[ProcessInDB])
async def list_processes(
    skip: int = 0, 
    limit: int = 10, 
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """List the latest processes for the current user"""
    processes = db.query(Process).filter(
        Process.user_id == current_user_id
    ).order_by(Process.id.desc()).offset(skip).limit(limit).all()
    
    return processes

@router.get("/{process_id}", response_model=ProcessInDB)
async def get_process(
    process_id: int, 
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Get a specific process by ID"""
    process = db.query(Process).filter(
        Process.id == process_id,
        Process.user_id == current_user_id
    ).first()
    
    if not process:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Process with ID {process_id} not found"
        )
    
    return process