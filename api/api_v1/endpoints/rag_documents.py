from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging
from core.security import decrypt_api_key
from models.agent import Agent
from models.api_key import APIKey
from models.process import Process
from db.session import get_db
from agents import create_agent
from core.log_utils import create_log

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/{agent_id}/upload_document")
async def upload_document_to_rag_agent(
    agent_id: int,
    document: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user_id: int = None
):
    """Upload a document to an existing RAG agent
    
    Args:
        agent_id: ID of the RAG agent to use
        document: PDF document to upload to the agent's knowledge base
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
        
        # Process only the document (not a query)
        context = {
            "document_content": document_content,
            "filename": document.filename,
            "process_id": process.id
        }
        
        # We'll pass an empty query to just process the document
        result = await agent_instance.process("", context)
        
        create_log(
            db=db,
            user_id=current_user_id,
            log_type=3,  # Custom log type for RAG
            description=f"Successfully uploaded document '{document.filename}' to RAG agent '{db_agent.name}'"
        )
        
        return {
            "status": "success",
            "message": f"Document '{document.filename}' successfully uploaded to RAG agent",
            "document_info": result.get("document", {})
        }
        
    except Exception as e:
        logger.error(f"Error uploading document to RAG agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )
