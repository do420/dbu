"""
Comprehensive tests for extended mini services endpoints.
Tests chat-generate, audio management, update/delete operations, and file upload functionality.
"""
import pytest
import os
import json
import tempfile
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open
from fastapi import HTTPException, status, UploadFile
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from io import BytesIO

# Import the modules to test
from api.api_v1.endpoints.mini_services import (
    chat_generate_mini_service,
    list_service_audio_files,
    get_audio_by_process,
    update_mini_service,
    delete_mini_service,
    upload_file
)
from models.mini_service import MiniService
from models.agent import Agent
from models.process import Process
from models.user import User
from models.api_key import APIKey
from schemas.mini_service import MiniServiceCreate
from core.security import decrypt_api_key


class TestChatGenerateEndpoint:
    """Test comprehensive chat-generate mini service creation system"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def mock_user_with_api_key(self):
        """Mock user with stored Gemini API key"""
        user = Mock(spec=User)
        user.id = 1
        
        api_key_obj = Mock(spec=APIKey)
        api_key_obj.provider = "gemini"
        api_key_obj.api_key = "encrypted_key"
        user.api_keys = [api_key_obj]
        
        return user
    
    @pytest.fixture
    def valid_chat_request(self):
        """Valid chat request data"""
        return {
            "message": "I want to create a service that translates text to Spanish",
            "conversation_history": [],
            "gemini_api_key": "test_api_key"
        }
    
    @pytest.mark.asyncio
    async def test_chat_generate_no_user_id(self, mock_db, valid_chat_request):
        """Test chat generation without user ID"""
        with pytest.raises(HTTPException) as exc_info:
            await chat_generate_mini_service(
                chat_request=valid_chat_request,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "current_user_id parameter is required" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_chat_generate_no_message(self, mock_db):
        """Test chat generation without message"""
        with pytest.raises(HTTPException) as exc_info:
            await chat_generate_mini_service(
                chat_request={},
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Message is required" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_chat_generate_success_initial_requirements(self, mock_db, mock_user_with_api_key, valid_chat_request):
        """Test successful chat generation for initial requirements gathering"""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user_with_api_key
        
        # Mock Gemini API responses
        mock_requirements_response = Mock()
        mock_requirements_response.text = json.dumps({
            "checklist": {
                "service_purpose": {"completed": True, "value": "text translation"},
                "input_type": {"completed": True, "value": "text"},
                "output_type": {"completed": True, "value": "text"},
                "service_name": {"completed": False, "value": "TBD"}
            },
            "missing_items": ["service_name"],
            "all_requirements_complete": False
        })
        
        mock_user_response = Mock()
        mock_user_response.text = json.dumps({
            "message": "Great! I can see you want a text translation service. What would you like to name this service?"
        })
        
        with patch('core.security.decrypt_api_key', return_value="decrypted_key"), \
             patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model = Mock()
            mock_model.generate_content.side_effect = [mock_requirements_response, mock_user_response]
            mock_model_class.return_value = mock_model
            
            result = await chat_generate_mini_service(
                chat_request=valid_chat_request,
                db=mock_db,
                current_user_id=1
            )
        
        assert result["type"] == "chat_response"
        assert "translation service" in result["message"]
        assert "checklist" in result
        assert result["checklist"]["service_purpose"]["completed"] is True
        assert result["checklist"]["service_name"]["completed"] is False


class TestAudioFileEndpoints:
    """Test audio file management endpoints"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def mock_mini_service(self):
        """Mock mini service"""
        service = Mock(spec=MiniService)
        service.id = 1
        service.owner_id = 1
        service.name = "Test Service"
        return service
    
    @pytest.fixture
    def mock_processes(self):
        """Mock processes with audio files"""
        processes = []
        for i in range(3):
            process = Mock(spec=Process)
            process.id = i + 1
            process.mini_service_id = 1
            process.created_at = "2024-01-01T00:00:00"
            processes.append(process)
        return processes
    
    @pytest.mark.asyncio
    async def test_list_service_audio_files_success(self, mock_db, mock_mini_service, mock_processes):
        """Test successful audio file listing"""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_mini_service
        mock_db.query.return_value.filter.return_value.all.return_value = mock_processes
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024):
            
            result = await list_service_audio_files(
                service_id=1,
                db=mock_db,
                current_user_id=1
            )
        
        assert len(result) == 3
        for i, audio_file in enumerate(result):
            assert audio_file["process_id"] == i + 1
            assert audio_file["audio_url"] == f"/audio/process_{i + 1}.mp3"
            assert audio_file["file_size"] == 1024
    
    @pytest.mark.asyncio
    async def test_list_service_audio_files_no_user_id(self, mock_db):
        """Test audio file listing without user ID"""
        with pytest.raises(HTTPException) as exc_info:
            await list_service_audio_files(
                service_id=1,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_get_audio_by_process_success(self, mock_db):
        """Test successful audio file retrieval by process ID"""
        mock_process = Mock(spec=Process)
        mock_process.id = 1
        mock_db.query.return_value.filter.return_value.first.return_value = mock_process
        
        with patch('os.path.exists', return_value=True), \
             patch('fastapi.responses.FileResponse') as mock_file_response:
            
            result = await get_audio_by_process(
                process_id=1,
                db=mock_db
            )
        
        mock_file_response.assert_called_once()
        call_args = mock_file_response.call_args
        assert "process_1.mp3" in call_args.kwargs["path"]
        assert call_args.kwargs["media_type"] == "audio/mpeg"
        assert call_args.kwargs["filename"] == "process_1.mp3"


class TestMiniServiceUpdateDelete:
    """Test mini service update and delete operations"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def mock_mini_service(self):
        """Mock mini service for updates"""
        service = Mock(spec=MiniService)
        service.id = 1
        service.owner_id = 1
        service.name = "Original Service"
        service.description = "Original description"
        service.workflow = {"nodes": {"0": {"agent_id": 1, "next": None}}}
        service.input_type = "text"
        service.output_type = "text"
        return service
    
    @pytest.fixture
    def update_data(self):
        """Update data for mini service"""
        return MiniServiceCreate(
            name="Updated Service",
            description="Updated description",
            workflow={"nodes": {"0": {"agent_id": 1, "next": None}}},
            input_type="text",
            output_type="text"
        )
    
    @pytest.mark.asyncio
    async def test_update_mini_service_success(self, mock_db, mock_mini_service, update_data):
        """Test successful mini service update"""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_mini_service
        mock_db.commit = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            result = await update_mini_service(
                service_id=1,
                mini_service_update=update_data,
                db=mock_db,
                current_user_id=1
            )
        
        assert result["message"] == "Mini service updated successfully"
        assert result["mini_service"]["name"] == "Updated Service"
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_mini_service_no_user_id(self, mock_db, update_data):
        """Test mini service update without user ID"""
        with pytest.raises(HTTPException) as exc_info:
            await update_mini_service(
                service_id=1,
                mini_service_update=update_data,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_delete_mini_service_success(self, mock_db):
        """Test successful mini service deletion"""
        mock_service = Mock(spec=MiniService)
        mock_service.id = 1
        mock_service.name = "Test Service"
        mock_service.owner_id = 1
        
        # Mock the execute calls for the new SQLAlchemy style
        mock_execute_result = Mock()
        mock_execute_result.scalar_one_or_none.return_value = mock_service
        
        # Mock the scalars for process deletion
        mock_scalars = Mock()
        mock_scalars.all.return_value = []  # No related processes
        
        mock_execute_result_2 = Mock()
        mock_execute_result_2.scalars.return_value = mock_scalars
        
        mock_db.execute.side_effect = [mock_execute_result, mock_execute_result_2]
        mock_db.delete = Mock()
        mock_db.commit = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            result = await delete_mini_service(
                service_id=1,
                db=mock_db,
                current_user_id=1
            )
        
        assert result["message"] == "Mini service deleted successfully"
        mock_db.delete.assert_called_once_with(mock_service)
        mock_db.commit.assert_called_once()


class TestFileUploadEndpoint:
    """Test file upload functionality"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def mock_upload_file(self):
        """Mock upload file"""
        file_content = b"test file content"
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.mp3"
        upload_file.content_type = "audio/mpeg"
        upload_file.read = AsyncMock(return_value=file_content)
        return upload_file
      @pytest.mark.asyncio
    async def test_upload_file_success(self, mock_db, mock_upload_file):
        """Test successful file upload"""
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('os.makedirs'), \
             patch('uuid.uuid4', return_value=Mock(hex='test-uuid')), \
             patch('api.api_v1.endpoints.mini_services.create_log'):
            
            result = await upload_file(
                file=mock_upload_file,
                db=mock_db,
                current_user_id=1
            )
        
        assert result["filename"] == "test.mp3"
        assert result["saved_as"] == "test-uuid.mp3"
        assert result["size"] == len(b"test file content")
        assert result["content_type"] == "audio/mpeg"
        mock_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_file_no_user_id(self, mock_db, mock_upload_file):
        """Test file upload without user ID"""
        with pytest.raises(HTTPException) as exc_info:
            await upload_file(
                file=mock_upload_file,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_upload_file_invalid_type(self, mock_db):
        """Test upload of invalid file type"""
        invalid_file = Mock(spec=UploadFile)
        invalid_file.filename = "test.exe"
        invalid_file.content_type = "application/x-executable"
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_file(
                file=invalid_file,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid file type" in str(exc_info.value.detail)
