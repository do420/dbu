import os
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from sqlalchemy.orm import Session
from agents.transribe_agent import TranscribeAgent
from core.log_utils import create_log
from db.session import get_db
from models.agent import Agent
from models.api_key import APIKey
from models.process import Process
from models.favorite_agent import FavoriteAgent
from schemas.agent import AgentCreate, AgentInDB
from schemas.favorite_agent import FavoriteAgentCreate, FavoriteAgentInDB, FavoriteAgentCountResponse
from agents import create_agent
from core.security import decrypt_api_key
import logging
from fastapi.responses import FileResponse
import shutil
import base64
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
            "input_type": "text",
            "output_type": "text",
            "api_key_required": "True"
        },
        {
            "type": "google_translate",
            "input_type": "text",
            "output_type": "text",
            "api_key_required": "False"
        },
        {
            "type": "claude",
            "input_type": "text",
            "output_type": "text",
            "api_key_required": "True"
        },
    ]
    return agent_types

@router.post("/", response_model=AgentInDB)
async def create_agent_endpoint(
    request: Request,
    db: Session = Depends(get_db),
    current_user_id: int = None,
    enhance_prompt: int = 0,
    file: UploadFile = File(None)
):
    """Create a new agent. For RAG agents, require a PDF and precompute ChromaDB collection. Accepts JSON for non-RAG, multipart for RAG."""
    import json
    content_type = request.headers.get("content-type", "")
    # Parse input depending on content type
    if content_type.startswith("application/json"):
        body = await request.json()
        name = body.get("name")
        system_instruction = body.get("system_instruction", "")
        agent_type = body.get("agent_type")
        input_type = body.get("input_type")
        output_type = body.get("output_type")
        config = body.get("config", {})
        # If config is not dict, try to parse
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except Exception:
                config = {}
    else:
        form = await request.form()
        name = form.get("name")
        system_instruction = form.get("system_instruction", "")
        agent_type = form.get("agent_type")
        input_type = form.get("input_type")
        output_type = form.get("output_type")
        config = form.get("config", "{}")
        try:
            config = json.loads(config) if isinstance(config, str) else config
        except Exception:
            config = {}
        # file is already handled by FastAPI param
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    valid_types = ["text", "image", "sound", "video", "document"]
    if input_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input_type. Must be one of: {valid_types}"
        )
    if output_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid output_type. Must be one of: {valid_types}"
        )
    # --- RAG agent special handling ---
    if agent_type and agent_type.lower() == "rag":
        if not file or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="RAG agents require a PDF file upload."
            )
        # Get Gemini API key from user's saved API keys
        gemini_api_key_obj = db.query(APIKey).filter(
            APIKey.user_id == current_user_id,
            APIKey.provider == "gemini"
        ).first()
        if not gemini_api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gemini API key is required for RAG agents but not found in your account."
            )
        gemini_api_key = decrypt_api_key(gemini_api_key_obj.api_key)
        # Prepare ChromaDB dir
        chroma_dir = os.path.join("db", "chroma", f"rag_collection_{name}")
        os.makedirs(chroma_dir, exist_ok=True)
        # Save PDF
        pdf_path = os.path.join(chroma_dir, file.filename)
        with open(pdf_path, "wb") as f_out:
            f_out.write(await file.read())
        # Process PDF and create ChromaDB
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        chroma_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_dir
        )
        _ = chroma_db.similarity_search("test", k=1)
    # --- ENHANCE SYSTEM PROMPT FEATURE (unchanged) ---
    enhanced_prompt = system_instruction
    is_enhanced = False
    if enhance_prompt:
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
        enhancement_request = (
            f"You are an expert prompt engineer. "
            f"Given the following agent details, enhance and improve the system prompt for optimal performance. "
            f"Agent Name: {name}\n"
            f"Description: {system_instruction}\n"
            f"Type: {agent_type}\n"
            f"Input Type: {input_type}\n"
            f"Output Type: {output_type}\n"
            f"Original System Prompt: {system_instruction}\n\n"
            f"Return ONLY the improved system prompt."
        )
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        try:
            response = model.generate_content(enhancement_request)
            enhanced_prompt = response.text.strip() if hasattr(response, "text") else str(response)
            is_enhanced = True
            create_log(
                db=db,
                user_id=current_user_id,
                log_type=2,
                description=f"Enhanced system prompt for agent '{name}' using Gemini."
            )
        except Exception as e:
            logger.error(f"Failed to enhance system prompt: {str(e)}")
            create_log(
                db=db,
                user_id=current_user_id,
                log_type=2,
                description=f"Failed to enhance system prompt for agent '{name}': {str(e)}"
            )
            enhanced_prompt = system_instruction
    # --- Create agent in database ---
    db_agent = Agent(
        name=name,
        system_instruction=enhanced_prompt,
        agent_type=agent_type,
        config=config,
        input_type=input_type,
        output_type=output_type,
        owner_id=current_user_id,
        is_enhanced=is_enhanced
    )
    create_log(
        db=db,
        user_id=current_user_id,
        log_type=1,
        description=f"Created agent '{name}'"
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
    if current_user_id is not None:
        # For RAG agents, only return those owned by the current user
        # For other agent types, return all
        agents = db.query(Agent).filter(
            (Agent.agent_type != "rag") | 
            ((Agent.agent_type == "rag") & (Agent.owner_id == current_user_id))
        ).offset(skip).limit(limit).all()
    else:        # If no user ID provided, only return non-RAG agents
        agents = db.query(Agent).filter(
            Agent.agent_type != "rag"
        ).offset(skip).limit(limit).all()
    return agents

@router.get("/favorites", response_model=List[AgentInDB])
async def get_favorite_agents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Get user's favorite agents"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Get favorite agents with join
    favorite_agents = db.query(Agent).join(
        FavoriteAgent,
        Agent.id == FavoriteAgent.agent_id
    ).filter(
        FavoriteAgent.user_id == current_user_id
    ).order_by(
        FavoriteAgent.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    return favorite_agents

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
    
    # Check if the agent is used in any mini-service by querying the MiniService table and checking if the agent_id is in the workflow
    from models.mini_service import MiniService
    mini_service_usage = db.query(MiniService).filter(
        MiniService.workflow.contains({"agent_id": agent_id})
    ).first()
    # If the agent is used in a mini-service, raise an error
   
   
    
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
        # For RAG, do NOT require or pass document_content/filename, just query
        result = await agent_instance.process(input_text, context)

        # DEBUG: If this is a RAG agent, log and return extra info
        if db_agent.agent_type.lower() == "rag":
            # Try to include debug info from the agent if available
            debug_info = result.copy()
            # If the agent returns context or chunks, include them
            if "debug_context" in result:
                debug_info["debug_context"] = result["debug_context"]
            # Log the result for inspection
            logger.info(f"RAG DEBUG: Query='{input_text}' | Result={result}")
            # Also return the debug info in the API response
            return debug_info
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


@router.post("/test-tts", response_class=FileResponse)
async def test_tts_configuration_frontend(test_data: Dict[str, Any]):
    """Test TTS configuration with a sample text and return audio file (frontend endpoint)"""
    from agents.tts_agent import TTSAgent
    import tempfile
    import os
    
    try:
        # Get test parameters from nested config structure
        config_data = test_data.get("config", {})
        voice = config_data.get("voice", "en-US-ChristopherNeural")
        test_text = test_data.get("text", "Hello, this is a test of the text-to-speech configuration.")
        rate = config_data.get("rate", "+0%")
        volume = config_data.get("volume", "+0%")
        pitch = config_data.get("pitch", "+0Hz")
        
        # Create a temporary TTS agent with test configuration
        config = {
            "voice": voice,
            "rate": rate,
            "volume": volume,
            "pitch": pitch
        }
        
        agent_instance = TTSAgent(config, "")
        
        # Create context with temporary file handling
        context = {
            "filename": "tts_test_audio.mp3",
            "process_id": "test"
        }
        
        # Test the TTS conversion
        result = await agent_instance.process(test_text, context)
        
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
                detail="Failed to generate test audio file"
            )
        
        # Return the audio file
        filename = f"tts_test_{voice}.mp3"
        return FileResponse(
            path=audio_file_path,
            media_type="audio/mpeg",
            filename=filename,
            headers={
                "X-Voice-Used": result.get("voice", voice),
                "X-Voice-Info": str(result.get("voice_info", {})),
                "X-Text-Length": str(result.get("text_length", len(test_text)))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS configuration test failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS configuration test failed: {str(e)}"
        )


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
       
        raise
    except Exception as e:
        # Clean up the process record
       
        
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
       
        raise
    except Exception as e:
        # Clean up the process record
       
        
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


# @router.post("/{agent_id}/transcribe", response_model=Dict[str, Any])
# async def transcribe_media(
#     agent_id: int,
#     db: Session = Depends(get_db),
#     current_user_id: int = 1,  # Replace with actual user ID from authentication
#     file: UploadFile = File(...),
#     language: str = Form("en"),
#     include_timestamps: bool = Form(False)
# ):
#     """
#     Transcribe an uploaded media file
    
#     Parameters:
#     - file: The audio/video file to transcribe
#     - language: Language code (optional, defaults to 'en')
#     - include_timestamps: Whether to include timestamps in the output (optional)
#     """
#     # Get the agent
#     db_agent = db.query(Agent).filter(
#         Agent.id == agent_id,
#     ).first()
    
#     if not db_agent:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Agent with ID {agent_id} not found"
#         )
    
#     # Verify this is a transcribe agent
#     if db_agent.agent_type not in ["transcribe", "whisperx"]:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Agent with ID {agent_id} is not a transcription agent"
#         )
    
   
    
#     # Save the uploaded file to _INPUT directory
#     filename = f"upload_{process.id}_{file.filename}"
#     input_dir = "_INPUT"
#     os.makedirs(input_dir, exist_ok=True)
#     file_path = os.path.join(input_dir, filename)
    
#     try:
#         # Save uploaded file
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#     except Exception as e:
#         db.delete(process)
#         db.commit()
#         logger.error(f"Error saving uploaded file: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error saving uploaded file: {str(e)}"
#         )
#     finally:
#         # Reset file position
#         await file.seek(0)
    
#     # Update agent config with additional parameters if provided
#     config = db_agent.config.copy()
#     if include_timestamps is not None:
#         config["include_timestamps"] = include_timestamps
    
#     # Create agent instance
#     try:
#         agent_instance = create_agent(
#             db_agent.agent_type, 
#             config, 
#             db_agent.system_instruction
#         )
#     except Exception as e:
#         # Clean up the process record and file
#         db.delete(process)
#         db.commit()
#         try:
#             os.remove(file_path)
#         except:
#             pass
        
#         logger.error(f"Failed to initialize transcription agent {agent_id}: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to initialize transcription agent {agent_id}: {str(e)}"
#         )
    
#     # Process with the agent
#     try:
#         # Add process_id to context
#         context = {"language": language}
        
#         # Prepare input data
#         transcribe_input = {
#             "file_path": file_path,
#             "language": language
#         }
        
#         # Call the agent with the file path
#         result = await agent_instance.process(transcribe_input, context)
        
#         if "error" in result:
#             # Clean up the process record
        
            
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=result["error"]
#             )
        
      
       
        
#         # Add the original filename to the result
#         result["original_filename"] = file.filename
        
#         db.commit()
        
#         return result
#     except HTTPException:
#         # Clean up the file
#         try:
#             os.remove(file_path)
#         except:
#             pass
#         raise
#     except Exception as e:
#         # Clean up the process record and file
      
#         try:
#             os.remove(file_path)
#         except:
#             pass
        
#         logger.error(f"Error during transcription: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error during transcription: {str(e)}"
#         )
    

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
    
  
   
    
    # Create agent instance
    try:
        agent_instance = create_agent(
            db_agent.agent_type, 
            db_agent.config, 
            db_agent.system_instruction
        )
    except Exception as e:
        # Clean up the process record
       
        
        logger.error(f"Failed to initialize GoogleTranslate agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize GoogleTranslate agent {agent_id}: {str(e)}"
        )
    
    # Process with the agent
    try:
        context = input_data.get("context", {})
        
        
        result = await agent_instance.process(input_text, context)
        
        if "error" in result:
            # Clean up the process record
           
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
    
       
        
        db.commit()
        
        return result
    except Exception as e:
        # Clean up the process record
       
        logger.error(f"Error during translation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during translation: {str(e)}"
        )
    
@router.post("/{agent_id}/run/rag_document")
async def run_rag_agent_with_document(
    agent_id: int,
    input: str = Form(...),
    api_key: str = Form(...),  # Direct API key string
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Run a RAG agent with a query using the precomputed ChromaDB collection."""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    if not input or not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'input' and 'api_key' are required in the form body."
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
    try:
     
     
        # Use the persistent directory for ChromaDB per agent (by name or id)
        chroma_dir = os.path.join("db", "chroma", f"rag_collection_{db_agent.name}")
        if not os.path.exists(chroma_dir):
            # Try fallback to id-based dir for backward compatibility
            chroma_dir = os.path.join("db", "chroma", f"rag_collection_{agent_id}")
        if not os.path.exists(chroma_dir):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"ChromaDB collection for agent {db_agent.name} not found."
            )
        # Load ChromaDB collection
        from langchain_community.vectorstores import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        chroma_db = Chroma(
            embedding_function=embeddings,
            persist_directory=chroma_dir
        )
        # Retrieve relevant document chunks for the input
        relevant_chunks = chroma_db.similarity_search(input, k=5)
        source_documents = [chunk.page_content for chunk in relevant_chunks]
        source_text = "\n\n".join(source_documents)
        # Create the Gemini prompt using input and sources
        prompt = f"""Based on the following context from the document, please answer this question:\n\nQUESTION: {input}\n\nDOCUMENT CONTEXT:\n{source_text}\n\nAnswer the question based only on the information provided in the document context.\n"""
        # Use Gemini to generate a response
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        # Prepare the result
        result = {
            "answer": response.text,
            "source_documents": source_documents[:3],  # Include top 3 sources
          
        }
        # Update process with token usage estimate
       
        db.commit()
        return result
    except Exception as e:
        logger.error(f"Error processing RAG query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing RAG query: {str(e)}"
        )
# ============ FAVORITE AGENTS ENDPOINTS ============

@router.post("/{agent_id}/favorite", response_model=FavoriteAgentInDB)
async def add_favorite_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Add an agent to user's favorites"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Check if agent exists
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    # Check if already favorited
    existing_favorite = db.query(FavoriteAgent).filter(
        FavoriteAgent.user_id == current_user_id,
        FavoriteAgent.agent_id == agent_id
    ).first()
    
    if existing_favorite:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is already in favorites"
        )
    
    # Create favorite record
    favorite = FavoriteAgent(
        user_id=current_user_id,
        agent_id=agent_id
    )
    
    db.add(favorite)
    db.commit()
    db.refresh(favorite)
    
    create_log(
        db=db,
        user_id=current_user_id,
        log_type=1,
        description=f"Added agent '{agent.name}' to favorites"
    )
    
    return favorite


@router.delete("/{agent_id}/favorite", status_code=status.HTTP_204_NO_CONTENT)
async def remove_favorite_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Remove an agent from user's favorites"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Find the favorite record
    favorite = db.query(FavoriteAgent).filter(
        FavoriteAgent.user_id == current_user_id,
        FavoriteAgent.agent_id == agent_id
    ).first()
    
    if not favorite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent is not in favorites"
        )
    
    # Get agent name for logging
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    agent_name = agent.name if agent else f"Agent {agent_id}"
    
    db.delete(favorite)
    db.commit()
    
    create_log(
        db=db,
        user_id=current_user_id,
        log_type=3,
        description=f"Removed agent '{agent_name}' from favorites"
    )
    return None


@router.get("/{agent_id}/favorite/status")
async def check_favorite_status(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Check if an agent is in user's favorites"""
    if current_user_id is None:
        return {"is_favorite": False}
    
    favorite = db.query(FavoriteAgent).filter(
        FavoriteAgent.user_id == current_user_id,
        FavoriteAgent.agent_id == agent_id
    ).first()
    
    return {"is_favorite": favorite is not None}


@router.get("/{agent_id}/favorite/count", response_model=FavoriteAgentCountResponse)
async def get_agent_favorite_count(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """Get the total number of users who have favorited this agent"""
    count = db.query(FavoriteAgent).filter(
        FavoriteAgent.agent_id == agent_id
    ).count()
    
    return FavoriteAgentCountResponse(
        agent_id=agent_id,
        favorite_count=count
    )
