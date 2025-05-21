import os
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from agents.transribe_agent import TranscribeAgent
from core.log_utils import create_log
from db.session import get_db
from models.agent import Agent
from models.api_key import APIKey
from models.process import Process
from schemas.agent import AgentCreate, AgentInDB
from agents import create_agent
from core.security import decrypt_api_key
import logging
from fastapi.responses import FileResponse
import shutil
#test
logger = logging.getLogger(__name__)
router = APIRouter()



@router.get("/types", response_model=List[Dict[str, str]])
async def get_agent_types():
    """Get a list of available agent types with their input and output types"""
    agent_types = [
        {
            "type": "gemini",
            "input_type": "text",
            "output_type": "text",
            "api_key_required": "True"
        },
        {
            "type": "openai",
            "input_type": "text",
            "output_type": "text",
            "api_key_required": "True"
        },
        {
            "type": "edge_tts",
            "input_type": "text",
            "output_type": "sound",
            "api_key_required": "False"
        },
        {
            "type": "bark_tts",
            "input_type": "text",
            "output_type": "sound",
             "api_key_required": "False"
        },
        {
            "type": "transcribe",
            "input_type": "sound",
            "output_type": "text",
            "api_key_required": "False"
        },
        {
            "type": "gemini_text2image",
            "input_type": "text",
            "output_type": "image",
            "api_key_required": "True"
        },
        {
            "type": "internet_research",
            "input_type": "text",
            "output_type": "text",
            "api_key_required": "False"
        },
        {
            "type": "document_parser",
            "input_type": "document",
            "output_type": "text",
            "api_key_required": "False"
        },        {
            "type": "custom_endpoint_llm",
            "input_type": "text",
            "output_type": "text",
            "api_key_required": "True"
        },
        {
            "type": "rag",
            "input_type": "document",
            "output_type": "text",
            "api_key_required": "True"
        },
        {
            "type": "google_translate",
            "input_type": "text",
            "output_type": "text",
            "api_key_required": "False"
        },
    ]
    return agent_types

@router.post("/", response_model=AgentInDB)
async def create_agent_endpoint(
    agent: AgentCreate, 
    db: Session = Depends(get_db),
    current_user_id: int = None,
    enhance_prompt: int = 0  # 0 or 1, or use bool if preferred
):
    """Create a new agent, optionally enhancing the system prompt using Gemini."""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    # Validate agent input and output types
    valid_types = ["text", "image", "sound", "video", "document"]
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

    
    is_enhanced = False
    # --- ENHANCE SYSTEM PROMPT FEATURE ---
    enhanced_prompt = agent.system_instruction
    if enhance_prompt:
        # Get user's Gemini API key
        gemini_api_key_obj = db.query(APIKey).filter(
            APIKey.user_id == current_user_id,
            APIKey.provider == "gemini"
        ).first()
        if not gemini_api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No Gemini API key found for prompt enhancement."
            )
        gemini_api_key = decrypt_api_key(gemini_api_key_obj.api_key)
        # Prepare enhancement prompt
        enhancement_request = (
            f"You are an expert prompt engineer. "
            f"Given the following agent details, enhance and improve the system prompt for optimal performance. "
            f"Agent Name: {agent.name}\n"
            f"Description: {agent.system_instruction}\n"
            f"Type: {agent.agent_type}\n"
            f"Input Type: {agent.input_type}\n"
            f"Output Type: {agent.output_type}\n"
            f"Original System Prompt: {agent.system_instruction}\n\n"
            f"Return ONLY the improved system prompt."
        )
        # Use Gemini to enhance the prompt
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        try:
            response = model.generate_content(enhancement_request)
            enhanced_prompt = response.text.strip() if hasattr(response, "text") else str(response)
            is_enhanced = True
            # Log enhancement
            create_log(
                db=db,
                user_id=current_user_id,
                log_type=2,
                description=f"Enhanced system prompt for agent '{agent.name}' using Gemini."
            )
        except Exception as e:
            logger.error(f"Failed to enhance system prompt: {str(e)}")
            create_log(
                db=db,
                user_id=current_user_id,
                log_type=2,
                description=f"Failed to enhance system prompt for agent '{agent.name}': {str(e)}"
            )
            # Optionally, you can raise or fallback to original prompt
            enhanced_prompt = agent.system_instruction

    # Create agent in database
    db_agent = Agent(
        name=agent.name,
        system_instruction=enhanced_prompt,
        agent_type=agent.agent_type,
        config=agent.config,
        input_type=agent.input_type,
        output_type=agent.output_type,
        owner_id=current_user_id,
        is_enhanced=is_enhanced
    )
    create_log(
        db=db,
        user_id=current_user_id,
        log_type=1,  
        description=f"Created agent '{agent.name}'"
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
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """List all agents owned by the current user"""
    #if current_user_id is None:
    #    raise HTTPException(
    #        status_code=status.HTTP_401_UNAUTHORIZED,
    #        detail="current_user_id parameter is required"
    #    )
    #.filter(
    #    Agent.owner_id == current_user_id
    #)
    agents = db.query(Agent).offset(skip).limit(limit).all()
    return agents

@router.get("/{agent_id}", response_model=AgentInDB)
async def get_agent(
    agent_id: int, 
    db: Session = Depends(get_db),
    current_user_id: int = None # Replace with actual user ID from authentication
):
    """Get a specific agent by ID"""
    #if current_user_id is None:
    #    raise HTTPException(
    #        status_code=status.HTTP_401_UNAUTHORIZED,
    #        detail="current_user_id parameter is required"
    #    )
    agent = db.query(Agent).filter(
        Agent.id == agent_id,
       #Agent.owner_id == current_user_id
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
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Update an existing agent"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
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

    create_log(
        db=db,
        user_id=current_user_id,
        log_type=1,  # 0: info
        description=f"Updated agent '{agent_update.name}'"
    )
    
    db.commit()
    db.refresh(db_agent)
    
    return db_agent
@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Delete an agent"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.owner_id == current_user_id
    ).first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    # Check if the agent is used in any mini-service
    mini_service_usage = db.query(Process).filter(
        Process.mini_service_id == agent_id
    ).first()
    
    if mini_service_usage:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with ID {agent_id} is used in a mini-service and cannot be deleted"
        )
    


    create_log(
        db=db,
        user_id=current_user_id,
        log_type=3,  
        description=f"Deleted agent '{db_agent.name}'"
    )

    db.delete(db_agent)
    db.commit()
    
    
    return None

@router.post("/{agent_id}/run")
async def run_agent(
    agent_id: int,
    input_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Run a single agent with the given input"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    # Get the agent
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id,
        #Agent.owner_id == current_user_id
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
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Generate speech from text using a TTS agent"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
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
    

@router.get("/voices/bark", response_model=List[str])
async def list_bark_voices():
    """List all available Bark TTS voice presets"""
    from agents.bark_agent import BarkTTSAgent
    voices = BarkTTSAgent.get_available_voices()
    return voices

@router.post("/{agent_id}/bark-tts", response_class=FileResponse)
async def generate_bark_speech(
    agent_id: int,
    input_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Generate speech from text using the Bark TTS agent"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
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
    
    # Verify this is a Bark TTS agent
    if db_agent.agent_type not in ["bark_tts", "suno-bark"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with ID {agent_id} is not a Bark TTS agent"
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
        mini_service_id=-1  # Direct agent use, not part of a service
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
        logger.error(f"Failed to initialize Bark TTS agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize Bark TTS agent {agent_id}: {str(e)}"
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
            media_type="audio/wav",
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
        
        logger.error(f"Error processing with Bark TTS agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing with Bark TTS agent: {str(e)}"
        )
    


@router.get("/models/transcribe", response_model=List[str])
async def list_transcribe_models():
    """List all available WhisperX transcription models"""

    models = TranscribeAgent.supported_models()
    return models

@router.get("/languages/transcribe", response_model=Dict[str, str])
async def list_transcribe_languages():
    """List all supported languages for transcription"""

    languages = TranscribeAgent.supported_languages()
    return languages


@router.post("/{agent_id}/transcribe", response_model=Dict[str, Any])
async def transcribe_media(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = 1,  # Replace with actual user ID from authentication
    file: UploadFile = File(...),
    language: str = Form("en"),
    include_timestamps: bool = Form(False)
):
    """
    Transcribe an uploaded media file
    
    Parameters:
    - file: The audio/video file to transcribe
    - language: Language code (optional, defaults to 'en')
    - include_timestamps: Whether to include timestamps in the output (optional)
    """
    # Get the agent
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id,
    ).first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    # Verify this is a transcribe agent
    if db_agent.agent_type not in ["transcribe", "whisperx"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with ID {agent_id} is not a transcription agent"
        )
    
    # Create a process record for this operation
    process = Process(
        user_id=current_user_id,
        total_tokens={},  # No tokens for transcription
        mini_service_id=-1  # Direct agent use, not part of a service
    )
    db.add(process)
    db.commit()
    db.refresh(process)
    
    # Save the uploaded file to _INPUT directory
    filename = f"upload_{process.id}_{file.filename}"
    input_dir = "_INPUT"
    os.makedirs(input_dir, exist_ok=True)
    file_path = os.path.join(input_dir, filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        db.delete(process)
        db.commit()
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving uploaded file: {str(e)}"
        )
    finally:
        # Reset file position
        await file.seek(0)
    
    # Update agent config with additional parameters if provided
    config = db_agent.config.copy()
    if include_timestamps is not None:
        config["include_timestamps"] = include_timestamps
    
    # Create agent instance
    try:
        agent_instance = create_agent(
            db_agent.agent_type, 
            config, 
            db_agent.system_instruction
        )
    except Exception as e:
        # Clean up the process record and file
        db.delete(process)
        db.commit()
        try:
            os.remove(file_path)
        except:
            pass
        
        logger.error(f"Failed to initialize transcription agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize transcription agent {agent_id}: {str(e)}"
        )
    
    # Process with the agent
    try:
        # Add process_id to context
        context = {"process_id": process.id, "language": language}
        
        # Prepare input data
        transcribe_input = {
            "file_path": file_path,
            "language": language
        }
        
        # Call the agent with the file path
        result = await agent_instance.process(transcribe_input, context)
        
        if "error" in result:
            # Clean up the process record
            db.delete(process)
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        # Add process_id to the result
        result["process_id"] = process.id
        
        # Add the original filename to the result
        result["original_filename"] = file.filename
        
        db.commit()
        
        return result
    except HTTPException:
        # Clean up the file
        try:
            os.remove(file_path)
        except:
            pass
        raise
    except Exception as e:
        # Clean up the process record and file
        db.delete(process)
        db.commit()
        try:
            os.remove(file_path)
        except:
            pass
        
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during transcription: {str(e)}"
        )
    

@router.post("/{agent_id}/run/image")
async def run_image_agent(
    agent_id: int,
    input_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Run a Gemini text-to-image agent with the given prompt."""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    # Get the agent
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id,
        #Agent.owner_id == current_user_id
    ).first()

    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )

    if db_agent.agent_type.lower() != "gemini_text2image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with ID {agent_id} is not a Gemini text-to-image agent"
        )

    # Extract the image generation prompt
    prompt = input_data.get("prompt", "")
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image generation prompt is required"
        )

    # Create the GeminiImageGeneration agent instance
    try:
        config = db_agent.config.copy()
        if "api_key" not in config:
            api_key = db.query(APIKey).filter(
                APIKey.user_id == current_user_id,
                APIKey.provider == "gemini" # API key provider is still "gemini"
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
        logger.error(f"Failed to initialize Gemini image agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize Gemini image agent {agent_id}: {str(e)}"
        )

    # Process the image generation prompt
    try:
        context = input_data.get("context", {})
        result = await agent_instance.process(prompt, context)
        return result
    except Exception as e:
        logger.error(f"Error processing image generation with agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image generation with agent: {str(e)}"
        )

@router.get("/languages/translate", response_model=Dict[str, str])
async def list_translate_languages():
    """List all supported languages for translation"""
    from agents.google_translate_agent import GoogleTranslateAgent
    languages = GoogleTranslateAgent.get_supported_languages()
    return languages


@router.post("/{agent_id}/translate")
async def translate_text(
    agent_id: int,
    input_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user_id: int = None  # Replace with actual user ID from authentication
):
    """Translate text using the Google Translate agent
    
    Parameters:
    - input: Text to translate
    - context: Optional context with target_language
    """
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Get the agent
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id,
    ).first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    # Verify this is a google_translate agent
    if db_agent.agent_type != "google_translate":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with ID {agent_id} is not a Google Translate agent"
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
        mini_service_id=-1  # Direct agent use, not part of a service
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
        # Clean up the process record
        db.delete(process)
        db.commit()
        
        logger.error(f"Failed to initialize GoogleTranslate agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize GoogleTranslate agent {agent_id}: {str(e)}"
        )
    
    # Process with the agent
    try:
        context = input_data.get("context", {})
        context["process_id"] = process.id
        
        result = await agent_instance.process(input_text, context)
        
        if "error" in result:
            # Clean up the process record
            db.delete(process)
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        # Add process_id to the result
        result["process_id"] = process.id
        
        db.commit()
        
        return result
    except Exception as e:
        # Clean up the process record
        db.delete(process)
        db.commit()
        
        logger.error(f"Error during translation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during translation: {str(e)}"
        )

@router.post("/{agent_id}/run/rag_document")
async def run_rag_agent_with_document(
    agent_id: int,
    query: str = Form(...),
    document: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Run a RAG agent with a query and uploaded document
    
    Args:
        agent_id: ID of the RAG agent to use
        query: User query to answer from the document
        document: PDF document to process
    """
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Get the agent
    db_agent = db.query(Agent).filter(
        Agent.id == agent_id
    ).first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    if db_agent.agent_type.lower() != "rag":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with ID {agent_id} is not a RAG agent"
        )
    
    # Get API key for Gemini
    api_key_obj = db.query(APIKey).filter(
        APIKey.user_id == current_user_id,
        APIKey.provider == "gemini"
    ).first()
    
    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini API key is required for RAG agent"
        )
    
    api_key = decrypt_api_key(api_key_obj.api_key)
    
    # Validate file type
    if not document.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported"
        )
    
    try:
        # Read document content
        document_content = await document.read()
        
        # Create process record
        process = Process(
            mini_service_id=None,  # Not associated with a mini service
            user_id=current_user_id,
            total_tokens={}  # Will update after processing
        )
        db.add(process)
        db.commit()
        db.refresh(process)
        
        # Create agent instance with API key
        config = db_agent.config.copy() if db_agent.config else {}
        config["api_key"] = api_key
        
        agent_instance = create_agent(
            db_agent.agent_type,
            config,
            db_agent.system_instruction
        )
        
        # Process the document and query
        context = {
            "document_content": document_content,
            "filename": document.filename,
            "process_id": process.id
        }
        
        result = await agent_instance.process(query, context)
        
        # Update the process record with token usage
        if "token_usage" in result:
            process.total_tokens = result["token_usage"]
            db.commit()
        
        create_log(
            db=db,
            user_id=current_user_id,
            log_type=3,  # Custom log type for RAG
            description=f"Successfully processed document '{document.filename}' with RAG agent '{db_agent.name}'"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing document with RAG agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )
