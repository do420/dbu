import os
from typing import List, Dict, Any, Optional
import uuid
import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
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
from models.user import User
from models.api_key import APIKey
from sqlalchemy.orm.attributes import flag_modified
import google.generativeai as genai
from core.security import decrypt_api_key

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
        for key, value in token_usage.items():
            prev_avg = mini_service.average_token_usage.get(key, None)
            if prev_avg is not None and mini_service.run_time > 1:
                mini_service.average_token_usage[key] = (
                    (prev_avg * (mini_service.run_time - 1) + value) / mini_service.run_time
                )
            else:
                mini_service.average_token_usage[key] = value
        flag_modified(mini_service, "average_token_usage")
        
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
    
    mini_services = db.query(
        MiniService, User.username
    ).join(
        User, MiniService.owner_id == User.id
    ).filter(
        (MiniService.is_public == True) | 
        (MiniService.owner_id == current_user_id)
    ).offset(skip).limit(limit).all()
    
    # Add owner_username to each mini service
    result = []
    for mini_service, username in mini_services:
        mini_service_dict = mini_service.__dict__.copy()
        mini_service_dict["owner_username"] = username

        # Extract agent IDs from workflow
        workflow = mini_service_dict.get("workflow", {})
        agent_ids = set()
        for node in workflow.get("nodes", {}).values():
            agent_id = node.get("agent_id")
            if agent_id is not None:
                agent_ids.add(int(agent_id))

        # Query agent types
        agent_types = set()
        if agent_ids:
            agents = db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
            agent_types = set(a.agent_type.lower() for a in agents)

        # Determine if any agent is external
        external_types = {"gemini", "openai"}
        tts_types = {"transcribe", "tts", "bark_tts", "edge_tts"}
        if agent_types and not (agent_types & external_types):
            # Only TTS/transcribe agents, show token usage as "-"
            mini_service_dict["average_token_usage"] = {key: "-" for key in mini_service_dict.get("average_token_usage", {})}
        # else: leave as is (JSON/dict)

        result.append(mini_service_dict)
    
    return result

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



@router.post("/upload")
async def upload_file(
    file: UploadFile,
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Upload a file for processing"""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    # Check file size (limit to 200MB)
    max_size = 200 * 1024 * 1024  # 200MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 200MB limit"
        )
    
    # Create uploads directory if it doesn't exist
    upload_dir = "_INPUT"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    try:
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        create_log(
            db=db,
            user_id=current_user_id,
            log_type=0,
            description=f"Uploaded file: {file.filename} ({len(file_content)} bytes)"
        )
        
        return {
            "filename": file.filename,
            "saved_as": unique_filename,
            "size": len(file_content),
            "content_type": file.content_type,
            "file_path": file_path
        }
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

@router.post("/chat-generate", response_model=Dict[str, Any])
async def chat_generate_mini_service(
    chat_request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Interactive chat with Gemini to generate a new mini service with agents
    
    Expected input:
    {
        "message": "User's message to Gemini",
        "conversation_history": [
            {"role": "user", "content": "previous message"},
            {"role": "assistant", "content": "previous response"}
        ],
        "create_service": false,  # Set to true when ready to create the service
        "gemini_api_key": "Optional Gemini API key"
    }
    """
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="current_user_id parameter is required"
        )
    
    message = chat_request.get("message")
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message is required"
        )
    
    conversation_history = chat_request.get("conversation_history", [])
    create_service = chat_request.get("create_service", False)
    
    # Get Gemini API key
    gemini_api_key = chat_request.get("gemini_api_key")
    if not gemini_api_key:
        # Try to get from user's stored API keys
        user = db.query(User).filter(User.id == current_user_id).first()
        if user and user.api_keys:
            # Find the Gemini API key from the user's stored API keys
            for api_key_obj in user.api_keys:
                if api_key_obj.provider.lower() == "gemini":
                    # Decrypt the stored API key
                    gemini_api_key = decrypt_api_key(api_key_obj.api_key)
                    break
    
    if not gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini API key is required. Please provide it in the request or save it in your profile."
        )
    
    try:
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Build conversation context
        conversation_text = ""
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        
        if create_service:
            # User wants to create the service - ask Gemini to generate the final specification
            system_prompt = f"""
Based on our conversation, please generate a complete mini-service specification. The user has confirmed they want to create the service.

Previous conversation:
{conversation_text}

Current user message: {message}

Please respond with a JSON object in the following exact format:

{{
    "service": {{
        "name": "Generated service name",
        "description": "Detailed description of what this service does",
        "input_type": "text|image|sound",
        "output_type": "text|image|sound",
        "is_public": false
    }},
    "agents": [
        {{
            "name": "Agent Name",
            "agent_type": "gemini|openai|claude|edge_tts|bark_tts|transcribe|gemini_text2image|internet_research|document_parser|google_translate",
            "system_instruction": "Detailed system prompt for the agent",
            "config": {{
                "model": "model_name_if_needed",
                "temperature": 0.7,
                "max_tokens": 1000
            }},
            "input_type": "text|image|sound",
            "output_type": "text|image|sound"
        }}
    ],
    "workflow": {{
        "nodes": {{
            "0": {{
                "agent_id": 0,
                "next": 1
            }},
            "1": {{
                "agent_id": 1,
                "next": null
            }}
        }}
    }}
}}

CRITICAL REQUIREMENTS:
- Always start workflow with node "0"
- agent_id in workflow should correspond to the index of agents in the agents array (0, 1, 2...)
- Use appropriate agent types from the available list
- For gemini agents, use model names like "gemini-pro", "gemini-pro-vision"
- For openai agents, use models like "gpt-3.5-turbo", "gpt-4"
- Make system instructions specific and detailed
- Ensure the workflow makes logical sense
- The last node should have "next": null
- Keep agent configs realistic and appropriate
- input_type/output_type must be "text", "image", or "sound" (not "file" or "audio")
- Each agent must have ALL required fields: name, agent_type, system_instruction, config, input_type, output_type

Respond ONLY with the JSON object, no other text. Do not wrap the JSON in markdown code blocks.
"""
            
            response = model.generate_content(system_prompt)
            response_text = response.text.strip()
            
            # Remove markdown code block formatting if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove ```
            response_text = response_text.strip()
            
            # Try to parse as JSON (service creation)
            try:
                generated_data = json.loads(response_text)
                
                # Validate the structure
                if not all(key in generated_data for key in ["service", "agents", "workflow"]):
                    raise ValueError("Invalid response structure")
                
                # Create agents first
                created_agents = []
                agent_id_mapping = {}
                
                for idx, agent_data in enumerate(generated_data["agents"]):
                    db_agent = Agent(
                        name=agent_data["name"],
                        agent_type=agent_data["agent_type"],
                        system_instruction=agent_data["system_instruction"],
                        config=agent_data.get("config", {}),
                        input_type=agent_data.get("input_type", "text"),
                        output_type=agent_data.get("output_type", "text"),
                        owner_id=current_user_id,
                        is_enhanced=False
                    )
                    
                    db.add(db_agent)
                    db.commit()
                    db.refresh(db_agent)
                    
                    created_agents.append(db_agent)
                    agent_id_mapping[idx] = db_agent.id
                
                # Update workflow with actual agent IDs
                updated_workflow = {"nodes": {}}
                for node_id, node_data in generated_data["workflow"]["nodes"].items():
                    array_index = node_data["agent_id"]
                    actual_agent_id = agent_id_mapping[array_index]
                    
                    updated_workflow["nodes"][str(node_id)] = {
                        "agent_id": actual_agent_id,
                        "next": node_data["next"]
                    }
                
                # Check if any agent is enhanced
                is_enhanced = any(agent.is_enhanced for agent in created_agents)
                
                # Create the mini service
                service_data = generated_data["service"]
                
                db_mini_service = MiniService(
                    name=service_data["name"],
                    description=service_data["description"],
                    workflow=updated_workflow,
                    input_type=service_data.get("input_type", "text"),
                    output_type=service_data.get("output_type", "text"),
                    owner_id=current_user_id,
                    average_token_usage={},
                    run_time=0,
                    is_enhanced=is_enhanced,
                    is_public=service_data.get("is_public", False)
                )
                
                db.add(db_mini_service)
                db.commit()
                db.refresh(db_mini_service)
                
                # Log the creation
                create_log(
                    db=db,
                    user_id=current_user_id,
                    log_type=0,
                    description=f"Generated mini-service '{service_data['name']}' with {len(created_agents)} agents using AI chat"
                )
                
                return {
                    "type": "service_created",
                    "mini_service": {
                        "id": db_mini_service.id,
                        "name": db_mini_service.name,
                        "description": db_mini_service.description,
                        "workflow": db_mini_service.workflow,
                        "input_type": db_mini_service.input_type,
                        "output_type": db_mini_service.output_type,
                        "owner_id": db_mini_service.owner_id,
                        "created_at": db_mini_service.created_at.isoformat(),
                        "is_enhanced": db_mini_service.is_enhanced,
                        "is_public": db_mini_service.is_public
                    },
                    "agents": [
                        {
                            "id": agent.id,
                            "name": agent.name,
                            "agent_type": agent.agent_type,
                            "system_instruction": agent.system_instruction,
                            "config": agent.config,
                            "input_type": agent.input_type,
                            "output_type": agent.output_type,
                            "owner_id": agent.owner_id,
                            "created_at": agent.created_at.isoformat(),
                            "is_enhanced": agent.is_enhanced
                        } for agent in created_agents
                    ],
                    "message": f"Successfully created mini-service '{service_data['name']}' with {len(created_agents)} agents!",
                    "conversation_history": conversation_history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": f"Perfect! I've created your mini-service '{service_data['name']}' with {len(created_agents)} specialized agents. The service is now ready to use!"}
                    ]
                }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Failed to parse service creation response: {response_text}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create service. Please try rephrasing your request."
                )
        
        else:
            # Regular conversation - help user design the service and evaluate completeness
            system_prompt = f"""
You are a friendly AI assistant helping users create custom mini-services. Your job is to guide them through the process step by step using a systematic checklist approach.

CONVERSATION STYLE:
- Be conversational and friendly
- Ask ONE question at a time
- Keep responses short (2-3 sentences max)
- Avoid technical jargon
- Show progress by mentioning what's been determined and what's still needed

SERVICE CREATION CHECKLIST:
Review the conversation and check off completed items:

☐ SERVICE PURPOSE: What specific task will this service accomplish?
☐ INPUT TYPE: What will the user provide? (text, image, or sound)
☐ OUTPUT TYPE: What should the service return? (text, image, or sound)
☐ SERVICE NAME: A simple, clear name for the service
☐ AGENT SELECTION: Which agents are needed to accomplish the task?

AVAILABLE AGENTS (you choose based on their needs):
- gemini: General purpose AI text generation and analysis (text → text)
- openai: Advanced text generation (text → text) 
- claude: Text analysis and reasoning (text → text)
- edge_tts: Convert text to speech (text → sound)
- bark_tts: High-quality text to speech (text → sound)
- transcribe: Convert audio to text (sound → text)
- gemini_text2image: Create images from descriptions (text → image)
- internet_research: Search the web for information (text → text)
- document_parser: Extract text from documents (document → text)
- google_translate: Translate between languages (text → text)

Previous conversation:
{conversation_text}

Current user message: {message}

INSTRUCTIONS:
1. First, mentally review the checklist based on the conversation history
2. If ALL 5 checklist items are complete (✓), automatically create the service by responding ONLY with "CREATE_SERVICE:" + JSON
3. If any items are missing, ask ONE friendly question to gather the next missing piece of information
4. When asking questions, briefly mention what you already know to show progress
5. Do not use any agents besides the ones listed above
6. If the requested service cannot be created with available agents, politely explain why and suggest alternatives

EXAMPLE RESPONSES WHEN GATHERING INFO:
- "Great! I understand you want to [PURPOSE]. What type of input will you provide - text, image, or audio?"
- "Perfect! So far I know: [PURPOSE] with [INPUT_TYPE] input. What would you like to get back - text, image, or audio?"
- "Excellent! I have: [PURPOSE], [INPUT_TYPE] → [OUTPUT_TYPE]. What would you like to name this service?"

When creating service, use this exact format:
CREATE_SERVICE: {{"service": {{"name": "Service Name", "description": "What it does", "input_type": "text|image|sound", "output_type": "text|image|sound", "is_public": false}}, "agents": [{{"name": "Tool Name", "agent_type": "tool_type", "system_instruction": "What this tool should do", "config": {{"temperature": 0.7, "model": "gemini-1.5-flash"}}, "input_type": "text|image|sound", "output_type": "text|image|sound"}}], "workflow": {{"nodes": {{"0": {{"agent_id": 0, "next": null}}}}}}}}
"""
            
            response = model.generate_content(system_prompt)
            response_text = response.text.strip()
            
            # Check if Gemini decided to create the service
            if response_text.startswith("CREATE_SERVICE:"):
                json_text = response_text[15:].strip()  # Remove "CREATE_SERVICE:" prefix
                
                # Remove markdown code block formatting if present
                if json_text.startswith("```json"):
                    json_text = json_text[7:]  # Remove ```json
                if json_text.endswith("```"):
                    json_text = json_text[:-3]  # Remove ```
                json_text = json_text.strip()
                
                try:
                    generated_data = json.loads(json_text)
                    
                    # Validate the structure
                    if not all(key in generated_data for key in ["service", "agents", "workflow"]):
                        raise ValueError("Invalid response structure")
                    
                    # Create agents first
                    created_agents = []
                    agent_id_mapping = {}
                    
                    for idx, agent_data in enumerate(generated_data["agents"]):
                        db_agent = Agent(
                            name=agent_data["name"],
                            agent_type=agent_data["agent_type"],
                            system_instruction=agent_data["system_instruction"],
                            config=agent_data.get("config", {}),
                            input_type=agent_data.get("input_type", "text"),
                            output_type=agent_data.get("output_type", "text"),
                            owner_id=current_user_id,
                            is_enhanced=False
                        )
                        
                        db.add(db_agent)
                        db.commit()
                        db.refresh(db_agent)
                        
                        created_agents.append(db_agent)
                        agent_id_mapping[idx] = db_agent.id
                    
                    # Update workflow with actual agent IDs
                    updated_workflow = {"nodes": {}}
                    for node_id, node_data in generated_data["workflow"]["nodes"].items():
                        array_index = node_data["agent_id"]
                        actual_agent_id = agent_id_mapping[array_index]
                        
                        updated_workflow["nodes"][str(node_id)] = {
                            "agent_id": actual_agent_id,
                            "next": node_data["next"]
                        }
                    
                    # Check if any agent is enhanced
                    is_enhanced = any(agent.is_enhanced for agent in created_agents)
                    
                    # Create the mini service
                    service_data = generated_data["service"]
                    
                    db_mini_service = MiniService(
                        name=service_data["name"],
                        description=service_data["description"],
                        workflow=updated_workflow,
                        input_type=service_data.get("input_type", "text"),
                        output_type=service_data.get("output_type", "text"),
                        owner_id=current_user_id,
                        average_token_usage={},
                        run_time=0,
                        is_enhanced=is_enhanced,
                        is_public=service_data.get("is_public", False)
                    )
                    
                    db.add(db_mini_service)
                    db.commit()
                    db.refresh(db_mini_service)
                    
                    # Log the creation
                    create_log(
                        db=db,
                        user_id=current_user_id,
                        log_type=0,
                        description=f"AI auto-generated mini-service '{service_data['name']}' with {len(created_agents)} agents"
                    )
                    
                    return {
                        "type": "service_created",
                        "mini_service": {
                            "id": db_mini_service.id,
                            "name": db_mini_service.name,
                            "description": db_mini_service.description,
                            "workflow": db_mini_service.workflow,
                            "input_type": db_mini_service.input_type,
                            "output_type": db_mini_service.output_type,
                            "owner_id": db_mini_service.owner_id,
                            "created_at": db_mini_service.created_at.isoformat(),
                            "is_enhanced": db_mini_service.is_enhanced,
                            "is_public": db_mini_service.is_public
                        },
                        "agents": [
                            {
                                "id": agent.id,
                                "name": agent.name,
                                "agent_type": agent.agent_type,
                                "system_instruction": agent.system_instruction,
                                "config": agent.config,
                                "input_type": agent.input_type,
                                "output_type": agent.output_type,
                                "owner_id": agent.owner_id,
                                "created_at": agent.created_at.isoformat(),
                                "is_enhanced": agent.is_enhanced
                            } for agent in created_agents
                        ],
                        "message": f"🎉 Great! I've automatically created your mini-service '{service_data['name']}' with {len(created_agents)} specialized agents. All requirements were satisfied and the service is ready to use!",
                        "conversation_history": conversation_history + [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": f"Perfect! I've analyzed our conversation and determined that all requirements are satisfied. I've automatically created your mini-service '{service_data['name']}' with {len(created_agents)} specialized agents. The service is now ready to use!"}
                        ]
                    }
                    
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.error(f"Failed to parse auto-generated service: {json_text}")
                    # Fall back to regular conversation if JSON parsing fails
                    return {
                        "type": "chat_response",
                        "message": "I think we have enough information to create your service, but let me ask a few more clarifying questions to make sure everything is perfect.",
                        "conversation_history": conversation_history + [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": "I think we have enough information to create your service, but let me ask a few more clarifying questions to make sure everything is perfect."}
                        ]
                    }
            
            else:
                # Regular conversation response
                return {
                    "type": "chat_response",
                    "message": response_text,
                    "conversation_history": conversation_history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": response_text}
                    ]
                }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(e)}"
        )
               