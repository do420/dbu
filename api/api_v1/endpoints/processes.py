from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.session import get_db
from models.process import Process
from schemas.process import ProcessInDB
from models.log import Log
from schemas.log import LogInDB

router = APIRouter()

@router.get("/", response_model=List[ProcessInDB])
async def list_processes(
    skip: int = 0, 
    limit: int = 10, 
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """List the latest processes for the current user"""

    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    processes = db.query(Process).filter(
        Process.user_id == current_user_id
    ).order_by(Process.id.desc()).offset(skip).limit(limit).all()
    
    return processes

@router.get("/recent-activities", response_model=List[LogInDB])
async def get_recent_logs(
    limit: int = 5,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    print("Fetching recent logs for user:", current_user_id)

    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )

    try:
        logs = db.query(Log).filter(
            Log.user_id == current_user_id
        ).order_by(Log.created_at.desc()).limit(limit).all()

        print("Fetched logs:", logs)
        return logs
    except Exception as e:
        print("Error while fetching logs:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{process_id}", response_model=ProcessInDB)
async def get_process(
    process_id: int, 
    db: Session = Depends(get_db),
    current_user_id: int = None # Replace with actual user ID from authentication
):
    """Get a specific process by ID"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
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




