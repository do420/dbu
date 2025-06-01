from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_
from db.session import get_db
from models.process import Process
from models.mini_service import MiniService
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
    """List the latest processes for the current user with mini service details"""

    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Join processes with mini services and filter out processes without mini_service_id
    query_result = db.query(
        Process,
        MiniService.name.label('mini_service_name'),
        MiniService.description.label('mini_service_description'),
        MiniService.input_type.label('mini_service_input_type'),
        MiniService.output_type.label('mini_service_output_type'),
        MiniService.is_enhanced.label('mini_service_is_enhanced')
    ).join(
        MiniService, Process.mini_service_id == MiniService.id
    ).filter(
        and_(Process.user_id == current_user_id, Process.mini_service_id.isnot(None))
    ).order_by(Process.id.desc()).offset(skip).limit(limit).all()
    
    # Convert the result to the expected format
    processes = []
    for row in query_result:
        process = row[0]  # The Process object
        process_dict = {
            "id": process.id,
            "mini_service_id": process.mini_service_id,
            "user_id": process.user_id,
            "total_tokens": process.total_tokens,
            "created_at": process.created_at,
            "mini_service_name": row[1],
            "mini_service_description": row[2],
            "mini_service_input_type": row[3],
            "mini_service_output_type": row[4],
            "mini_service_is_enhanced": row[5]
        }
        processes.append(ProcessInDB(**process_dict))
    
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
    """Get a specific process by ID with mini service details"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Join process with mini service and filter out processes without mini_service_id
    query_result = db.query(
        Process,
        MiniService.name.label('mini_service_name'),
        MiniService.description.label('mini_service_description'),
        MiniService.input_type.label('mini_service_input_type'),
        MiniService.output_type.label('mini_service_output_type'),
        MiniService.is_enhanced.label('mini_service_is_enhanced')
    ).join(
        MiniService, Process.mini_service_id == MiniService.id
    ).filter(
        and_(
            Process.id == process_id, 
            Process.user_id == current_user_id,
            Process.mini_service_id.isnot(None)
        )
    ).first()
    
    if not query_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Process with ID {process_id} not found or has no associated mini service"
        )
    
    # Convert the result to the expected format
    process = query_result[0]  # The Process object
    process_dict = {
        "id": process.id,
        "mini_service_id": process.mini_service_id,
        "user_id": process.user_id,
        "total_tokens": process.total_tokens,
        "created_at": process.created_at,
        "mini_service_name": query_result[1],
        "mini_service_description": query_result[2],
        "mini_service_input_type": query_result[3],
        "mini_service_output_type": query_result[4],
        "mini_service_is_enhanced": query_result[5]
    }
    
    return ProcessInDB(**process_dict)




