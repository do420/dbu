import os
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.session import get_db
from models.agent import Agent
from models.api_key import APIKey
from models.process import Process
from schemas.agent import AgentCreate, AgentInDB
from agents import create_agent
from core.security import decrypt_api_key
import logging
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/", response_model=AgentInDB)
async def create_agent_endpoint(
    agent: AgentCreate, 
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Create a new agent"""
    # Validate agent input and output types
    valid_types = ["text", "image", "sound"]
    if agent.input_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input_type. Must be one of: {valid_types}"
        )
    
    if agent.output_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid output_type. Must be one of: {valid_types}"
        )
    
    # If agent uses an API key, verify it exists
    if agent.agent_type in ["gemini", "openai"] and "api_key" not in agent.config:
        # Try to load from user's API keys
        api_key = db.query(APIKey).filter(
            APIKey.user_id == current_user_id,
            APIKey.provider == agent.agent_type
        ).first()
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No API key found for {agent.agent_type}. Please add an API key first."
            )
        
        # Decrypt and add the API key to the config
        decrypted_key = decrypt_api_key(api_key.api_key)
        agent.config["api_key"] = decrypted_key
    
    # Create agent in database
    db_agent = Agent(
        name=agent.name,
        system_instruction=agent.system_instruction,
        agent_type=agent.agent_type,
        config=agent.config,
        input_type=agent.input_type,
        output_type=agent.output_type,
        owner_id=current_user_id
    )
    
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    
    return db_agent

@router.get("/", response_model=List[AgentInDB])
async def list_agents(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """List all agents owned by the current user"""
    agents = db.query(Agent).filter(
        Agent.owner_id == current_user_id
    ).offset(skip).limit(limit).all()
    return agents

@router.get("/{agent_id}", response_model=AgentInDB)
async def get_agent(
    agent_id: int, 
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Get a specific agent by ID"""
    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.owner_id == current_user_id
    ).first()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    return agent

@router.put("/{agent_id}", response_model=AgentInDB)
async def update_agent(
    agent_id: int,
    agent_update: AgentCreate,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Update an existing agent"""
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.owner_id == current_user_id
    ).first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    # Update agent attributes
    db_agent.name = agent_update.name
    db_agent.system_instruction = agent_update.system_instruction
    db_agent.config = agent_update.config
    db_agent.input_type = agent_update.input_type
    db_agent.output_type = agent_update.output_type
    
    db.commit()
    db.refresh(db_agent)
    
    return db_agent

@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Delete an agent"""
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.owner_id == current_user_id
    ).first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    db.delete(db_agent)
    db.commit()
    
    return None

@router.post("/{agent_id}/run")
async def run_agent(
    agent_id: int,
    input_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Run a single agent with the given input"""
    # Get the agent
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.owner_id == current_user_id
    ).first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    # Extract input text
    input_text = input_data.get("input", "")
    if not input_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text is required"
        )
    
    # Create agent instance
    try:
        # If agent uses API key but it's not in config, get from API keys
        config = db_agent.config.copy()
        if db_agent.agent_type in ["gemini", "openai"] and "api_key" not in config:
            api_key = db.query(APIKey).filter(
                APIKey.user_id == current_user_id,
                APIKey.provider == db_agent.agent_type
            ).first()
            
            if api_key:
                decrypted_key = decrypt_api_key(api_key.api_key)
                config["api_key"] = decrypted_key
        
        agent_instance = create_agent(
            db_agent.agent_type, 
            config, 
            db_agent.system_instruction
        )
    except Exception as e:
        logger.error(f"Failed to initialize agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize agent {agent_id}: {str(e)}"
        )
    
    # Process with the agent
    try:
        context = input_data.get("context", {})
        result = await agent_instance.process(input_text, context)
        return result
    except Exception as e:
        logger.error(f"Error processing with agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing with agent: {str(e)}"
        )
    
@router.get("/voices/tts", response_model=List[Dict[str, str]])
async def list_tts_voices():
    """List all available TTS voices"""
    from agents.tts_agent import TTSAgent
    voices = await TTSAgent.list_voices()
    return voices

@router.get("/languages/tts", response_model=List[str])
async def list_tts_languages():
    """List all supported TTS languages"""
    from agents.tts_agent import TTSAgent
    languages = await TTSAgent.get_supported_languages()
    return languages


@router.post("/{agent_id}/tts", response_class=FileResponse)
async def generate_speech(
    agent_id: int,
    input_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user_id: int = 1  # Replace with actual user ID from authentication
):
    """Generate speech from text using a TTS agent"""
    # Get the agent
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.owner_id == current_user_id
    ).first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    # Verify this is a TTS agent
    if db_agent.agent_type not in ["text2speech", "tts"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with ID {agent_id} is not a TTS agent"
        )
    
    # Extract input text
    input_text = input_data.get("input", "")
    if not input_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text is required"
        )
        
    # Create a process record for this operation
    process = Process(
        user_id=current_user_id,
        total_tokens={"total_tokens": len(input_text.split())},  # Approximate token count
        mini_service_id=-1 #stands for no mini-service, direct use
    )
    db.add(process)
    db.commit()
    db.refresh(process)
    
    # Create agent instance
    try:
        agent_instance = create_agent(
            db_agent.agent_type, 
            db_agent.config, 
            db_agent.system_instruction
        )
    except Exception as e:
        logger.error(f"Failed to initialize TTS agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize TTS agent {agent_id}: {str(e)}"
        )
    
    # Process with the agent
    try:
        context = input_data.get("context", {})
        # Add process_id to context
        context["process_id"] = process.id
        
        result = await agent_instance.process(input_text, context)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        # Get the path to the audio file
        audio_file_path = result.get("audio_file")
        
        if not audio_file_path or not os.path.exists(audio_file_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate audio file"
            )
        
        # Return the audio file
        filename = os.path.basename(audio_file_path)
        return FileResponse(
            path=audio_file_path,
            media_type="audio/mpeg",
            filename=filename
        )
    except HTTPException:
        # Clean up the process record
        db.delete(process)
        db.commit()
        raise
    except Exception as e:
        # Clean up the process record
        db.delete(process)
        db.commit()
        
        logger.error(f"Error processing with TTS agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing with TTS agent: {str(e)}"
        )