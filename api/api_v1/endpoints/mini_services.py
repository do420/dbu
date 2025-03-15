from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.session import get_db
from models.mini_service import MiniService
from models.agent import Agent
from models.process import Process
from schemas.mini_service import MiniServiceCreate, MiniServiceInDB
from agents import create_agent
from agents.multi_agent import WorkflowProcessor
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/", response_model=MiniServiceInDB)
async def create_mini_service(
    mini_service: MiniServiceCreate, 
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Create a new mini service"""
    # Validate workflow structure
    if "start_node" not in mini_service.workflow:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow must contain a 'start_node' field"
        )
    
    if "nodes" not in mini_service.workflow:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow must contain a 'nodes' dictionary"
        )
    
    # Validate that all nodes have an agent_id and that referenced agents exist
    agent_ids = set()
    for node_id, node in mini_service.workflow["nodes"].items():
        if "agent_id" not in node:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Node {node_id} must specify an agent_id"
            )
        agent_ids.add(int(node["agent_id"]))
    
    # Check that all referenced agents exist and belong to the current user
    for agent_id in agent_ids:
        agent = db.query(Agent).filter(
            Agent.id == agent_id,
            Agent.owner_id == current_user_id
        ).first()
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID {agent_id} not found or you don't have permission to use it"
            )
    
    # Create mini service in database
    db_mini_service = MiniService(
        name=mini_service.name,
        description=mini_service.description,
        workflow=mini_service.workflow,
        input_type=mini_service.input_type,
        output_type=mini_service.output_type,
        owner_id=current_user_id,
        average_token_usage={},
        run_time=0
    )
    
    db.add(db_mini_service)
    db.commit()
    db.refresh(db_mini_service)
    
    return db_mini_service

@router.get("/", response_model=List[MiniServiceInDB])
async def list_mini_services(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """List all mini services owned by the current user"""
    mini_services = db.query(MiniService).filter(
        MiniService.owner_id == current_user_id
    ).offset(skip).limit(limit).all()
    
    return mini_services

@router.get("/{service_id}", response_model=MiniServiceInDB)
async def get_mini_service(
    service_id: int, 
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Get a specific mini service by ID"""
    mini_service = db.query(MiniService).filter(
        MiniService.id == service_id,
        MiniService.owner_id == current_user_id
    ).first()
    
    if not mini_service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mini service with ID {service_id} not found"
        )
    
    return mini_service

@router.put("/{service_id}", response_model=MiniServiceInDB)
async def update_mini_service(
    service_id: int,
    mini_service_update: MiniServiceCreate,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Update an existing mini service"""
    db_mini_service = db.query(MiniService).filter(
        MiniService.id == service_id,
        MiniService.owner_id == current_user_id
    ).first()
    
    if not db_mini_service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mini service with ID {service_id} not found"
        )
    
    # Update mini service attributes
    db_mini_service.name = mini_service_update.name
    db_mini_service.description = mini_service_update.description
    db_mini_service.workflow = mini_service_update.workflow
    db_mini_service.input_type = mini_service_update.input_type
    db_mini_service.output_type = mini_service_update.output_type
    
    db.commit()
    db.refresh(db_mini_service)
    
    return db_mini_service

@router.delete("/{service_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mini_service(
    service_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Delete a mini service"""
    db_mini_service = db.query(MiniService).filter(
        MiniService.id == service_id,
        MiniService.owner_id == current_user_id
    ).first()
    
    if not db_mini_service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mini service with ID {service_id} not found"
        )
    
    db.delete(db_mini_service)
    db.commit()
    
    return None

@router.post("/{service_id}/run")
async def run_mini_service(
    service_id: int,
    input_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Run a mini service with the given input"""
    # Get the mini service
    mini_service = db.query(MiniService).filter(
        MiniService.id == service_id
    ).first()
    
    if not mini_service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mini service with ID {service_id} not found"
        )
    
    # Extract input
    input_value = input_data.get("input")
    if input_value is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input is required"
        )
    
    # Get all agents used in this mini service's workflow
    agent_ids = set()
    for node_id, node in mini_service.workflow.get("nodes", {}).items():
        agent_id = node.get("agent_id")
        if agent_id:
            agent_ids.add(int(agent_id))
    
    # Load agent instances
    agents = {}
    for agent_id in agent_ids:
        agent_record = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        # Create agent instance
        try:
            agent_instance = create_agent(
                agent_record.agent_type, 
                agent_record.config, 
                agent_record.system_instruction
            )
            agents[str(agent_id)] = agent_instance
        except Exception as e:
            logger.error(f"Failed to initialize agent {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize agent {agent_id}: {str(e)}"
            )
    
    # Create and run the workflow processor
    workflow_processor = WorkflowProcessor(agents, mini_service.workflow)
    try:
        context = input_data.get("context", {})
        result = await workflow_processor.process(input_value, context)
        
        # Record the process
        token_usage = result.get("token_usage", {"total_tokens": 0})
        process = Process(
            mini_service_id=mini_service.id,
            user_id=current_user_id,
            total_tokens=token_usage
        )
        db.add(process)
        
        # Update mini service stats
        mini_service.run_time += 1
        
        # Update average token usage
        if mini_service.average_token_usage:
            for key, value in token_usage.items():
                if key in mini_service.average_token_usage:
                    # Calculate new average
                    mini_service.average_token_usage[key] = (
                        (mini_service.average_token_usage[key] * (mini_service.run_time - 1) + value) 
                        / mini_service.run_time
                    )
                else:
                    mini_service.average_token_usage[key] = value
        else:
            mini_service.average_token_usage = token_usage
            
        db.commit()
        
        return result
    except Exception as e:
        logger.error(f"Error processing with mini service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing with mini service: {str(e)}"
        )