import os
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select
from core.log_utils import create_log
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
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Create a new mini service"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    # Validate workflow structure
    if "nodes" not in mini_service.workflow:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow must contain a 'nodes' dictionary"
        )
    
    # Check if node 0 exists - we'll use this as the start node
    if "0" not in mini_service.workflow["nodes"] and 0 not in mini_service.workflow["nodes"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow must contain a node with ID 0 as the start node"
        )
    
    # Validate the node structure and extract agent IDs
    agent_ids = set()
    node_ids = set()
    
    for node_id_str, node in mini_service.workflow["nodes"].items():
        # Convert node_id to int for consistent handling
        node_id = int(node_id_str) if isinstance(node_id_str, str) else node_id_str
        node_ids.add(node_id)
        
        # Check for required agent_id
        if "agent_id" not in node:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Node {node_id} is missing 'agent_id'"
            )
        
        agent_id = node.get("agent_id")
        if agent_id is not None:  # Allow for None agent_id if needed
            agent_ids.add(int(agent_id))
        
        # Validate "next" field if present
        if "next" in node and node["next"] is not None:
            next_node = node["next"]
            if not isinstance(next_node, int) and not (isinstance(next_node, str) and next_node.isdigit()):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Node {node_id} has invalid 'next' value. Must be an integer node ID or null."
                )
            
    is_enhanced = False
     # Check that the agent exists and belongs to the current user
     # Check that all referenced agent IDs exist
    for agent_id in agent_ids:
        #agent = db.query(Agent).filter(
         #   Agent.id == agent_id,
          #  Agent.owner_id == current_user_id
        #).first()
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                #detail=f"Agent with ID {agent_id} not found or you don't have permission to use it"
                detail=f"Agent with ID {agent_id} not found"
            )
        
        # Check if this agent has enhanced prompt
        if agent.is_enhanced:
            is_enhanced = True
           
    
    
    # Restructure the workflow to ensure node IDs are stored as strings
    # (This is for compatibility with JSON serialization in the database)
    standardized_workflow = {
        "nodes": {}
    }
    
    for node_id_str, node in mini_service.workflow["nodes"].items():
        node_id = str(node_id_str)  # Ensure node_id is a string
        standardized_workflow["nodes"][node_id] = {
            "agent_id": node["agent_id"],
            "next": node.get("next")
        }
    
    # Create mini service in database
    db_mini_service = MiniService(
        name=mini_service.name,
        description=mini_service.description,
        workflow=standardized_workflow,  # Use the standardized workflow
        input_type=mini_service.input_type,
        output_type=mini_service.output_type,
        owner_id=current_user_id,
        average_token_usage={},
        run_time=0,
        is_enhanced=is_enhanced,
        is_public= mini_service.is_public or False,  # Default to private if not specified
    )

    visibility_status = "public" if mini_service.is_public else "private"
    create_log(
        db=db,
        user_id=current_user_id,
        log_type=0,  # 0: info
        description=f"Created a new {visibility_status} mini-service: '{mini_service.name}'" + 
                    (f" (with enhanced prompts)" if is_enhanced else "")
    )

    
    db.add(db_mini_service)
    db.commit()
    db.refresh(db_mini_service)
    
    return db_mini_service

@router.post("/{service_id}/run")
async def run_mini_service(
    service_id: int,
    input_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Run a mini service with the given input
    
    Input data should include:
    - input: The input for the mini service
    - context: Optional context data
    - api_keys: Dictionary mapping agent IDs to API keys (e.g. {"0": "sk-...", "1": "..."})
    """
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    # Get the mini service
    mini_service = db.query(MiniService).filter(
        MiniService.id == service_id
    ).first()
    
    if not mini_service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mini service with ID {service_id} not found"
        )
    
    if mini_service.is_public is not True and mini_service.owner_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to run this mini service"
        )
    
    
    # Extract input and API keys
    input_value = input_data.get("input")
    api_keys = input_data.get("api_keys", {})

    if input_value is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input is required"
        )
    
    # Create process record first to get a process ID
    # This ensures we have a process ID before running the workflow
    process = Process(
        mini_service_id=mini_service.id,
        user_id=current_user_id,
        total_tokens={}  # Will update after processing
    )
    db.add(process)
    db.commit()
    db.refresh(process)
    
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
            db.delete(process)
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        # Create agent instance
        try:

            # Start with the agent's base configuration
            config = agent_record.config.copy() if agent_record.config else {}
            
            # Check if this agent requires an API key and one was provided
            agent_str_id = str(agent_id)
            print(f"Agent ID: {agent_str_id}, API Keys: {api_keys}")
            if agent_str_id in api_keys and api_keys[agent_str_id]:
                print(f"Using API key for agent {agent_str_id}: {api_keys[agent_str_id]}")
                # Use the API key provided for this specific agent
                config["api_key"] = api_keys[agent_str_id]

            agent_instance = create_agent(
                agent_record.agent_type, 
                config,
                agent_record.system_instruction
            )
            agents[str(agent_id)] = agent_instance
        except Exception as e:
            logger.error(f"Failed to initialize agent {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize agent {agent_id}: {str(e)}"
            )
    
    # Create and run the workflow processor with the updated structure
    # We need to modify the workflow to include a start_node field for compatibility
    workflow = mini_service.workflow.copy()
    workflow["start_node"] = "0"  # Always start with node 0
    
    workflow_processor = WorkflowProcessor(agents, workflow)
    try:
        # Add process_id to the context
        context = input_data.get("context", {})
        context["process_id"] = process.id
        
        # Run the workflow
        result = await workflow_processor.process(input_value, context)
        
        # Update the process record with token usage
        token_usage = result.get("token_usage", {"total_tokens": 0})
        process.total_tokens = token_usage
        
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
        
        # Add process_id to the result for easy reference
        result["process_id"] = process.id
        
        # If any TTS audio files were generated, include their URLs
        audio_urls = []
        for step_result in result.get("results", []):
            raw_data = step_result.get("raw", {})
            if raw_data and "audio_url" in raw_data:
                audio_urls.append({
                    "step": step_result.get("step"),
                    "audio_url": raw_data["audio_url"]
                })
        
        if audio_urls:
            result["audio_urls"] = audio_urls

        create_log(
            db=db,
            user_id=current_user_id,
            log_type=5,  # 0: info
            description=f"Succesfully run a mini-service: '{mini_service.name}'"
        )
            
        db.commit()
        
        return result
    except Exception as e:
        logger.error(f"Error processing with mini service: {str(e)}")
        # Clean up the process record if there was an error
        db.delete(process)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing with mini service: {str(e)}"
        )

@router.get("/", response_model=List[MiniServiceInDB])
async def list_mini_services(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """List all mini services owned by the current user"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Fetch mini services that are public or owned by the current user
    mini_services = db.query(MiniService).filter(
        (MiniService.is_public == True) | 
        (MiniService.owner_id == current_user_id)
    ).offset(skip).limit(limit).all()
    


    
    return mini_services

@router.get("/{service_id}", response_model=MiniServiceInDB)
async def get_mini_service(
    service_id: int, 
    db: Session = Depends(get_db),
    current_user_id: int = None # Replace with actual user ID from authentication
):
    """Get a specific mini service by ID"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    mini_service = db.query(MiniService).filter(
        MiniService.id == service_id
    ).first()
    
    if not mini_service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mini service with ID {service_id} not found"
        )
    
    if mini_service.is_public is not True and mini_service.owner_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this mini service"
        )
    
    return mini_service

@router.put("/{service_id}", response_model=MiniServiceInDB)
async def update_mini_service(
    service_id: int,
    mini_service_update: MiniServiceCreate,
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Update an existing mini service"""

    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
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
    current_user_id: int = None  # assume you wire your auth here
):
    if current_user_id is None:
        raise HTTPException(401, "current_user_id required")

    try:
        # Fetch & permission check
        mini_service = db.execute(
            select(MiniService).where(
                MiniService.id == service_id,
                MiniService.owner_id == current_user_id
            )
        ).scalar_one_or_none()
        if not mini_service:
            raise HTTPException(404, "Not found or no permission")

        # Delete related processes
        related = db.execute(
            select(Process).where(Process.mini_service_id == service_id)
        ).scalars().all()
        for p in related:
            db.delete(p)

        # Log and delete
        create_log(
            db=db,
            user_id=current_user_id,
            log_type=4,
            description=f"Deleted mini-service '{mini_service.name}'"
        )
        db.delete(mini_service)

        db.commit()   # single commit here
        return None

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting mini-service {service_id}: {e}")
        raise HTTPException(500, f"Failed to delete mini-service: {e}")


@router.get("/audio/{process_id}", response_class=FileResponse)
async def get_audio_by_process(
    process_id: int,
    db: Session = Depends(get_db)
):
    """Get audio file for a specific process"""
    # Verify the process exists
    process = db.query(Process).filter(Process.id == process_id).first()
    if not process:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Process with ID {process_id} not found"
        )
    
    # Construct the filename based on process ID
    filename = f"process_{process_id}.mp3"
    audio_path = os.path.join("_OUTPUT", filename)
    
    # Check if file exists
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file for process {process_id} not found"
        )
    
    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=filename
    )


@router.get("/{service_id}/audio", response_model=List[Dict[str, Any]])
async def list_service_audio_files(
    service_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """List all audio files generated by a mini service"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    # Check if mini service exists and belongs to the user
    mini_service = db.query(MiniService).filter(
        MiniService.id == service_id,
        MiniService.owner_id == current_user_id
    ).first()
    
    if not mini_service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mini service with ID {service_id} not found"
        )
    
    # Get all processes for this mini service
    processes = db.query(Process).filter(Process.mini_service_id == service_id).all()
    
    audio_files = []
    for process in processes:
        filename = f"process_{process.id}.mp3"
        audio_path = os.path.join("_OUTPUT", filename)
        
        if os.path.exists(audio_path):
            audio_files.append({
                "process_id": process.id,
                "created_at": process.created_at,
                "audio_url": f"/audio/{filename}",
                "file_size": os.path.getsize(audio_path)
            })
    
    return audio_files
