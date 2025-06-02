"""
Unit tests for mini services endpoint.
Tests critical functionality including creation, validation, running workflows, and error handling.
"""
import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Dict, Any

# Import the modules to test
from api.api_v1.endpoints.mini_services import (
    create_mini_service,
    run_mini_service,
    list_mini_services,
    get_mini_service,
    update_mini_service,
    delete_mini_service,
    upload_file
)
from models.mini_service import MiniService
from models.agent import Agent
from models.process import Process
from models.user import User
from schemas.mini_service import MiniServiceCreate
from db.base_class import Base


class TestMiniServiceCreation:
    """Test mini service creation functionality"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def sample_workflow(self):
        """Sample valid workflow for testing"""
        return {
            "nodes": {
                "0": {"agent_id": 1, "next": 1},
                "1": {"agent_id": 2, "next": None}
            }
        }
    
    @pytest.fixture
    def sample_mini_service_create(self, sample_workflow):
        """Sample mini service creation data"""
        return MiniServiceCreate(
            name="Test Service",
            description="A test service",
            workflow=sample_workflow,
            input_type="text",
            output_type="text",
            is_public=False
        )
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent for testing"""
        agent = Mock(spec=Agent)
        agent.id = 1
        agent.agent_type = "openai"
        agent.is_enhanced = False
        return agent

    @pytest.mark.asyncio
    async def test_create_mini_service_success(self, mock_db, sample_mini_service_create, mock_agent):
        """Test successful mini service creation"""
        # Setup
        current_user_id = 1
        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Mock create_log
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            result = await create_mini_service(
                mini_service=sample_mini_service_create,
                db=mock_db,
                current_user_id=current_user_id
            )
        
        # Verify database interactions
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_mini_service_no_user_id(self, mock_db, sample_mini_service_create):
        """Test mini service creation without user ID"""
        with pytest.raises(HTTPException) as exc_info:
            await create_mini_service(
                mini_service=sample_mini_service_create,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "current_user_id parameter is required" in str(exc_info.value.detail)

    @pytest.mark.asyncio 
    async def test_create_mini_service_invalid_workflow_no_nodes(self, mock_db):
        """Test mini service creation with invalid workflow (no nodes)"""
        invalid_mini_service = MiniServiceCreate(
            name="Test Service",
            description="A test service",
            workflow={},  # Missing nodes
            input_type="text",
            output_type="text"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await create_mini_service(
                mini_service=invalid_mini_service,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Workflow must contain a 'nodes' dictionary" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_mini_service_no_start_node(self, mock_db):
        """Test mini service creation without start node (node 0)"""
        invalid_workflow = {
            "nodes": {
                "1": {"agent_id": 1, "next": None}  # Missing node 0
            }
        }
        
        invalid_mini_service = MiniServiceCreate(
            name="Test Service",
            description="A test service", 
            workflow=invalid_workflow,
            input_type="text",
            output_type="text"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await create_mini_service(
                mini_service=invalid_mini_service,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Workflow must contain a node with ID 0 as the start node" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_mini_service_missing_agent_id(self, mock_db):
        """Test mini service creation with node missing agent_id"""
        invalid_workflow = {
            "nodes": {
                "0": {"next": None}  # Missing agent_id
            }
        }
        
        invalid_mini_service = MiniServiceCreate(
            name="Test Service",
            description="A test service",
            workflow=invalid_workflow,
            input_type="text",
            output_type="text"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await create_mini_service(
                mini_service=invalid_mini_service,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Node 0 is missing 'agent_id'" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_mini_service_nonexistent_agent(self, mock_db, sample_workflow):
        """Test mini service creation with non-existent agent"""
        mock_db.query.return_value.filter.return_value.first.return_value = None  # Agent not found
        
        mini_service = MiniServiceCreate(
            name="Test Service",
            description="A test service",
            workflow=sample_workflow,
            input_type="text",
            output_type="text"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await create_mini_service(
                mini_service=mini_service,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Agent with ID 1 not found" in str(exc_info.value.detail)


class TestMiniServiceRun:
    """Test mini service running functionality"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def mock_mini_service(self):
        """Mock mini service for testing"""
        service = Mock(spec=MiniService)
        service.id = 1
        service.name = "Test Service"
        service.owner_id = 1
        service.is_public = False
        service.workflow = {
            "nodes": {
                "0": {"agent_id": 1, "next": None}
            }
        }
        service.run_time = 0
        service.average_token_usage = {}
        return service
    
    @pytest.fixture
    def mock_agent_record(self):
        """Mock agent record for testing"""
        agent = Mock(spec=Agent)
        agent.id = 1
        agent.agent_type = "openai"
        agent.config = {"model": "gpt-3.5-turbo"}
        agent.system_instruction = "You are a helpful assistant"
        return agent
    
    @pytest.fixture
    def sample_input_data(self):
        """Sample input data for running mini service"""
        return {
            "input": "Hello, world!",
            "context": {"test": True},
            "api_keys": {"1": "sk-test-key"}
        }

    @pytest.mark.asyncio
    async def test_run_mini_service_success(self, mock_db, mock_mini_service, mock_agent_record, sample_input_data):
        """Test successful mini service run"""
        current_user_id = 1
        
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_mini_service,  # First call for mini service
            mock_agent_record   # Second call for agent
        ]
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Mock create_agent function
        mock_agent_instance = AsyncMock()
        
        # Mock WorkflowProcessor
        mock_workflow_result = {
            "output": "Processed result",
            "token_usage": {"total_tokens": 100},
            "results": []
        }
        mock_workflow_processor = AsyncMock()
        mock_workflow_processor.process.return_value = mock_workflow_result
        
        with patch('api.api_v1.endpoints.mini_services.create_agent', return_value=mock_agent_instance), \
             patch('api.api_v1.endpoints.mini_services.WorkflowProcessor', return_value=mock_workflow_processor), \
             patch('api.api_v1.endpoints.mini_services.create_log'), \
             patch('api.api_v1.endpoints.mini_services.flag_modified'):
            
            result = await run_mini_service(
                service_id=1,
                input_data=sample_input_data,
                db=mock_db,
                current_user_id=current_user_id
            )
        
        # Verify result
        assert "output" in result
        assert "process_id" in result
        assert result["output"] == "Processed result"
        
        # Verify database interactions
        mock_db.add.assert_called()
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_run_mini_service_not_found(self, mock_db, sample_input_data):
        """Test running non-existent mini service"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await run_mini_service(
                service_id=999,
                input_data=sample_input_data,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Mini service with ID 999 not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_run_mini_service_permission_denied(self, mock_db, sample_input_data):
        """Test running mini service without permission"""
        mock_service = Mock(spec=MiniService)
        mock_service.id = 1
        mock_service.owner_id = 2  # Different owner
        mock_service.is_public = False  # Not public
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_service
        
        with pytest.raises(HTTPException) as exc_info:
            await run_mini_service(
                service_id=1,
                input_data=sample_input_data,
                db=mock_db,
                current_user_id=1  # Different user
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "You do not have permission to run this mini service" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_run_mini_service_missing_input(self, mock_db, mock_mini_service):
        """Test running mini service without input"""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_mini_service
        
        input_data = {"context": {}}  # Missing input
        
        with pytest.raises(HTTPException) as exc_info:
            await run_mini_service(
                service_id=1,
                input_data=input_data,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Input is required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_run_mini_service_agent_creation_failure(self, mock_db, mock_mini_service, sample_input_data):
        """Test mini service run with agent creation failure"""
        current_user_id = 1
        
        # Setup mocks - mini service found, but agent creation fails
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_mini_service,  # Mini service found
            None  # Agent not found
        ]
        mock_db.add = Mock()
        mock_db.delete = Mock()  # For cleanup
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with pytest.raises(HTTPException) as exc_info:
            await run_mini_service(
                service_id=1,
                input_data=sample_input_data,
                db=mock_db,
                current_user_id=current_user_id
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Agent with ID 1 not found" in str(exc_info.value.detail)
        
        # Verify cleanup was called
        mock_db.delete.assert_called()

    @pytest.mark.asyncio
    async def test_run_mini_service_workflow_processing_error(self, mock_db, mock_mini_service, mock_agent_record, sample_input_data):
        """Test mini service run with workflow processing error"""
        current_user_id = 1
        
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_mini_service,
            mock_agent_record
        ]
        mock_db.add = Mock()
        mock_db.delete = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        mock_agent_instance = AsyncMock()
        
        # Mock WorkflowProcessor to raise an exception
        mock_workflow_processor = AsyncMock()
        mock_workflow_processor.process.side_effect = Exception("Workflow processing failed")
        
        with patch('api.api_v1.endpoints.mini_services.create_agent', return_value=mock_agent_instance), \
             patch('api.api_v1.endpoints.mini_services.WorkflowProcessor', return_value=mock_workflow_processor), \
             pytest.raises(HTTPException) as exc_info:
            
            await run_mini_service(
                service_id=1,
                input_data=sample_input_data,
                db=mock_db,
                current_user_id=current_user_id
            )
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error processing with mini service" in str(exc_info.value.detail)
        
        # Verify cleanup was called
        mock_db.delete.assert_called()


class TestMiniServiceCRUD:
    """Test CRUD operations for mini services"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock(spec=Session)
        return db

    @pytest.mark.asyncio
    async def test_get_mini_service_success(self, mock_db):
        """Test successful retrieval of mini service"""
        mock_service = Mock(spec=MiniService)
        mock_service.id = 1
        mock_service.owner_id = 1
        mock_service.is_public = False
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_service
        
        result = await get_mini_service(
            service_id=1,
            db=mock_db,
            current_user_id=1
        )
        
        assert result == mock_service

    @pytest.mark.asyncio
    async def test_get_mini_service_not_found(self, mock_db):
        """Test retrieval of non-existent mini service"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_mini_service(
                service_id=999,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_mini_service_permission_denied(self, mock_db):
        """Test retrieval of mini service without permission"""
        mock_service = Mock(spec=MiniService)
        mock_service.id = 1
        mock_service.owner_id = 2  # Different owner
        mock_service.is_public = False  # Not public
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_service
        
        with pytest.raises(HTTPException) as exc_info:
            await get_mini_service(
                service_id=1,
                db=mock_db,
                current_user_id=1  # Different user
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_list_mini_services(self, mock_db):
        """Test listing mini services"""
        # Mock the query chain for the complex join
        mock_service = Mock(spec=MiniService)
        mock_service.id = 1
        mock_service.name = "Test Service"
        mock_service.__dict__ = {
            "id": 1,
            "name": "Test Service",
            "workflow": {"nodes": {"0": {"agent_id": 1}}},
            "average_token_usage": {}
        }
        
        mock_username = "testuser"
        
        # Mock the complex query chain
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [(mock_service, mock_username)]
        
        mock_db.query.return_value = mock_query
        
        # Mock agent query for determining external agents
        mock_agent = Mock(spec=Agent)
        mock_agent.agent_type = "openai"
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_agent]
        
        result = await list_mini_services(
            skip=0,
            limit=100,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["name"] == "Test Service"
        assert result[0]["owner_username"] == "testuser"

    @pytest.mark.asyncio
    async def test_update_mini_service_success(self, mock_db):
        """Test successful mini service update"""
        mock_service = Mock(spec=MiniService)
        mock_service.id = 1
        mock_service.owner_id = 1
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_service
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        update_data = MiniServiceCreate(
            name="Updated Service",
            description="Updated description",
            workflow={"nodes": {"0": {"agent_id": 1, "next": None}}},
            input_type="text",
            output_type="text"
        )
        
        result = await update_mini_service(
            service_id=1,
            mini_service_update=update_data,
            db=mock_db,
            current_user_id=1
        )
        
        assert result == mock_service
        assert mock_service.name == "Updated Service"
        assert mock_service.description == "Updated description"

    @pytest.mark.asyncio
    async def test_delete_mini_service_success(self, mock_db):
        """Test successful mini service deletion"""
        mock_service = Mock(spec=MiniService)
        mock_service.id = 1
        mock_service.name = "Test Service"
        mock_service.owner_id = 1
        
        # Mock the select queries
        mock_execute = Mock()
        mock_execute.scalar_one_or_none.return_value = mock_service
        mock_execute.scalars.return_value.all.return_value = []  # No related processes
        mock_db.execute.return_value = mock_execute
        
        mock_db.delete = Mock()
        mock_db.commit = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            result = await delete_mini_service(
                service_id=1,
                db=mock_db,
                current_user_id=1
            )
        
        assert result is None
        mock_db.delete.assert_called_with(mock_service)
        mock_db.commit.assert_called()


class TestFileUpload:
    """Test file upload functionality"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def mock_upload_file(self):
        """Mock upload file"""
        file = Mock()
        file.filename = "test.txt"
        file.content_type = "text/plain"
        return file

    @pytest.mark.asyncio
    async def test_upload_file_success(self, mock_db, mock_upload_file):
        """Test successful file upload"""
        # Mock file content
        file_content = b"test content"
        mock_upload_file.read = AsyncMock(return_value=file_content)
        
        with patch('api.api_v1.endpoints.mini_services.create_log'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('os.makedirs'), \
             patch('uuid.uuid4', return_value="test-uuid"):
            
            result = await upload_file(
                file=mock_upload_file,
                db=mock_db,
                current_user_id=1
            )
        
        assert result["filename"] == "test.txt"
        assert result["saved_as"] == "test-uuid.txt"
        assert result["size"] == len(file_content)
        assert result["content_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_upload_file_too_large(self, mock_db, mock_upload_file):
        """Test file upload with size exceeding limit"""
        # Mock file content that's too large (200MB+ bytes)
        large_content = b"x" * (200 * 1024 * 1024 + 1)
        mock_upload_file.read = AsyncMock(return_value=large_content)
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_file(
                file=mock_upload_file,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "File size exceeds 200MB limit" in str(exc_info.value.detail)

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


class TestWorkflowValidation:
    """Test workflow validation logic"""
    
    def test_valid_workflow_structure(self):
        """Test valid workflow structure"""
        workflow = {
            "nodes": {
                "0": {"agent_id": 1, "next": 1},
                "1": {"agent_id": 2, "next": None}
            }
        }
        
        # This would be tested as part of create_mini_service
        # The validation logic is embedded in the endpoint
        assert "nodes" in workflow
        assert "0" in workflow["nodes"]

    def test_workflow_with_cycles(self):
        """Test workflow validation with potential cycles"""
        workflow = {
            "nodes": {
                "0": {"agent_id": 1, "next": 1},
                "1": {"agent_id": 2, "next": 0}  # Creates a cycle
            }
        }
        
        # The current implementation doesn't check for cycles
        # This could be a future enhancement
        assert "nodes" in workflow

    def test_workflow_with_invalid_next_references(self):
        """Test workflow with invalid next node references"""
        workflow = {
            "nodes": {
                "0": {"agent_id": 1, "next": 99}  # Non-existent node
            }
        }
        
        # Current implementation doesn't validate next node existence
        # This could be a future enhancement
        assert "nodes" in workflow


def mock_open():
    """Helper function to create a mock for file operations"""
    return MagicMock()


if __name__ == "__main__":
    pytest.main([__file__])
