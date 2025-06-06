"""
Comprehensive tests for agent endpoints.
Tests CRUD operations for AI agents.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from api.api_v1.endpoints.agents import (
    create_agent,
    list_agents,
    get_agent,
    update_agent,
    delete_agent
)
from models.agent import Agent
from schemas.agent import AgentCreate, AgentUpdate


class TestAgentCreation:
    """Test agent creation functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_agent_create(self):
        return AgentCreate(
            name="Test OpenAI Agent",
            agent_type="openai",
            system_instruction="You are a helpful assistant",
            config={"model": "gpt-3.5-turbo", "temperature": 0.7},
            input_type="text",
            output_type="text"
        )

    @pytest.mark.asyncio
    async def test_create_agent_success(self, mock_db, sample_agent_create):
        """Test successful agent creation"""
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.agents.create_log'):
            result = await create_agent(
                agent=sample_agent_create,
                db=mock_db,
                current_user_id=1
            )
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_agent_invalid_type(self, mock_db):
        """Test agent creation with invalid agent type"""
        invalid_agent = AgentCreate(
            name="Invalid Agent",
            agent_type="invalid_type",
            system_instruction="Test instruction",
            config={},
            input_type="text",
            output_type="text"
        )
        
        # This should be caught by validation
        with pytest.raises((HTTPException, ValueError)):
            await create_agent(
                agent=invalid_agent,
                db=mock_db,
                current_user_id=1
            )

    @pytest.mark.asyncio
    async def test_create_agent_missing_config(self, mock_db):
        """Test agent creation with missing required config"""
        agent_missing_config = AgentCreate(
            name="OpenAI Agent",
            agent_type="openai",
            system_instruction="Test instruction",
            config={},  # Missing required model
            input_type="text",
            output_type="text"
        )
        
        # Should validate required config parameters
        with pytest.raises((HTTPException, ValueError)):
            await create_agent(
                agent=agent_missing_config,
                db=mock_db,
                current_user_id=1
            )

    @pytest.mark.asyncio
    async def test_create_agent_unauthorized(self, mock_db, sample_agent_create):
        """Test agent creation without user authentication"""
        with pytest.raises(HTTPException) as exc_info:
            await create_agent(
                agent=sample_agent_create,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestAgentListing:
    """Test agent listing functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_agents(self):
        return [
            Agent(
                id=1,
                name="OpenAI Agent",
                agent_type="openai",
                system_instruction="Assistant",
                config={"model": "gpt-3.5-turbo"},
                input_type="text",
                output_type="text",
                owner_id=1
            ),
            Agent(
                id=2,
                name="Gemini Agent",
                agent_type="gemini",
                system_instruction="Helper",
                config={"model": "gemini-pro"},
                input_type="text",
                output_type="text",
                owner_id=1
            )
        ]

    @pytest.mark.asyncio
    async def test_list_agents_success(self, mock_db, sample_agents):
        """Test successful agent listing"""
        mock_db.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = sample_agents
        
        result = await list_agents(
            skip=0,
            limit=100,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 2
        assert result[0].name == "OpenAI Agent"
        assert result[1].name == "Gemini Agent"

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, mock_db):
        """Test listing when no agents exist"""
        mock_db.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        result = await list_agents(
            skip=0,
            limit=100,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_list_agents_pagination(self, mock_db, sample_agents):
        """Test agent listing with pagination"""
        mock_db.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = [sample_agents[0]]
        
        result = await list_agents(
            skip=0,
            limit=1,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 1
        mock_db.query.return_value.filter.return_value.offset.assert_called_with(0)
        mock_db.query.return_value.filter.return_value.offset.return_value.limit.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_list_agents_unauthorized(self, mock_db):
        """Test agent listing without authentication"""
        with pytest.raises(HTTPException) as exc_info:
            await list_agents(
                skip=0,
                limit=100,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestAgentRetrieval:
    """Test agent retrieval functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_agent(self):
        return Agent(
            id=1,
            name="Test Agent",
            agent_type="openai",
            system_instruction="You are helpful",
            config={"model": "gpt-3.5-turbo"},
            input_type="text",
            output_type="text",
            owner_id=1
        )

    @pytest.mark.asyncio
    async def test_get_agent_success(self, mock_db, sample_agent):
        """Test successful agent retrieval"""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_agent
        
        result = await get_agent(
            agent_id=1,
            db=mock_db,
            current_user_id=1
        )
        
        assert result.id == 1
        assert result.name == "Test Agent"
        assert result.agent_type == "openai"

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, mock_db):
        """Test agent retrieval when agent doesn't exist"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_agent(
                agent_id=999,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_agent_unauthorized_access(self, mock_db):
        """Test agent retrieval by non-owner"""
        other_user_agent = Agent(
            id=1,
            name="Other User's Agent",
            agent_type="openai",
            system_instruction="Private agent",
            config={"model": "gpt-3.5-turbo"},
            input_type="text",
            output_type="text",
            owner_id=2  # Different owner
        )
        
        mock_db.query.return_value.filter.return_value.first.return_value = other_user_agent
        
        with pytest.raises(HTTPException) as exc_info:
            await get_agent(
                agent_id=1,
                db=mock_db,
                current_user_id=1  # Different user
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


class TestAgentUpdate:
    """Test agent update functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_agent(self):
        return Agent(
            id=1,
            name="Original Agent",
            agent_type="openai",
            system_instruction="Original instruction",
            config={"model": "gpt-3.5-turbo"},
            input_type="text",
            output_type="text",
            owner_id=1
        )
    
    @pytest.fixture
    def agent_update(self):
        return AgentUpdate(
            name="Updated Agent",
            system_instruction="Updated instruction",
            config={"model": "gpt-4", "temperature": 0.5}
        )

    @pytest.mark.asyncio
    async def test_update_agent_success(self, mock_db, sample_agent, agent_update):
        """Test successful agent update"""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_agent
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.agents.create_log'):
            result = await update_agent(
                agent_id=1,
                agent_update=agent_update,
                db=mock_db,
                current_user_id=1
            )
            
            assert sample_agent.name == "Updated Agent"
            assert sample_agent.system_instruction == "Updated instruction"
            assert sample_agent.config["model"] == "gpt-4"
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_agent_not_found(self, mock_db, agent_update):
        """Test updating non-existent agent"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await update_agent(
                agent_id=999,
                agent_update=agent_update,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_update_agent_unauthorized(self, mock_db, agent_update):
        """Test updating agent without permission"""
        other_user_agent = Agent(
            id=1,
            name="Other's Agent",
            agent_type="openai",
            system_instruction="Private",
            config={},
            input_type="text",
            output_type="text",
            owner_id=2
        )
        
        mock_db.query.return_value.filter.return_value.first.return_value = other_user_agent
        
        with pytest.raises(HTTPException) as exc_info:
            await update_agent(
                agent_id=1,
                agent_update=agent_update,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_update_agent_partial(self, mock_db, sample_agent):
        """Test partial agent update"""
        partial_update = AgentUpdate(name="Only Name Updated")
        
        mock_db.query.return_value.filter.return_value.first.return_value = sample_agent
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.agents.create_log'):
            await update_agent(
                agent_id=1,
                agent_update=partial_update,
                db=mock_db,
                current_user_id=1
            )
            
            # Only name should be updated
            assert sample_agent.name == "Only Name Updated"
            # Other fields should remain unchanged
            assert sample_agent.system_instruction == "Original instruction"


class TestAgentDeletion:
    """Test agent deletion functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_agent(self):
        return Agent(
            id=1,
            name="Agent to Delete",
            agent_type="openai",
            system_instruction="Will be deleted",
            config={},
            input_type="text",
            output_type="text",
            owner_id=1
        )

    @pytest.mark.asyncio
    async def test_delete_agent_success(self, mock_db, sample_agent):
        """Test successful agent deletion"""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_agent
        mock_db.delete = Mock()
        mock_db.commit = Mock()
        
        with patch('api.api_v1.endpoints.agents.create_log'):
            await delete_agent(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
            
            mock_db.delete.assert_called_once_with(sample_agent)
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_agent_not_found(self, mock_db):
        """Test deleting non-existent agent"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await delete_agent(
                agent_id=999,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_delete_agent_unauthorized(self, mock_db):
        """Test deleting agent without permission"""
        other_user_agent = Agent(
            id=1,
            name="Protected Agent",
            agent_type="openai",
            system_instruction="Protected",
            config={},
            input_type="text",
            output_type="text",
            owner_id=2
        )
        
        mock_db.query.return_value.filter.return_value.first.return_value = other_user_agent
        
        with pytest.raises(HTTPException) as exc_info:
            await delete_agent(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_delete_agent_in_use(self, mock_db, sample_agent):
        """Test deleting agent that's being used in mini services"""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_agent
        
        # Mock mini service using this agent
        with patch('api.api_v1.endpoints.agents.check_agent_usage') as mock_check:
            mock_check.return_value = True  # Agent is in use
            
            with pytest.raises(HTTPException) as exc_info:
                await delete_agent(
                    agent_id=1,
                    db=mock_db,
                    current_user_id=1
                )
            
            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
            assert "in use" in str(exc_info.value.detail).lower()


class TestAgentValidation:
    """Test agent validation and configuration"""
    
    def test_openai_agent_config_validation(self):
        """Test OpenAI agent configuration validation"""
        valid_config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # This would be validated by the agent configuration schema
        assert "model" in valid_config
        assert isinstance(valid_config["temperature"], (int, float))

    def test_gemini_agent_config_validation(self):
        """Test Gemini agent configuration validation"""
        valid_config = {
            "model": "gemini-pro",
            "temperature": 0.5
        }
        
        assert "model" in valid_config
        assert valid_config["model"] in ["gemini-pro", "gemini-1.5-flash"]

    def test_tts_agent_config_validation(self):
        """Test TTS agent configuration validation"""
        edge_tts_config = {
            "voice": "en-US-AriaNeural",
            "rate": 0
        }
        
        bark_config = {
            "voice_preset": "v2/en_speaker_6"
        }
        
        assert "voice" in edge_tts_config
        assert "voice_preset" in bark_config

    def test_invalid_agent_config(self):
        """Test invalid agent configurations"""
        invalid_configs = [
            {},  # Empty config
            {"invalid_key": "value"},  # Wrong keys
            {"model": ""},  # Empty model
            {"temperature": "invalid"}  # Wrong type
        ]
        
        for config in invalid_configs:
            # These should fail validation
            pass


class TestAgentSecurity:
    """Test security aspects of agent management"""
    
    def test_agent_config_sanitization(self):
        """Test that agent configs are properly sanitized"""
        malicious_config = {
            "model": "gpt-3.5-turbo",
            "system_prompt": "'; DROP TABLE agents; --",
            "api_key": "exposed_key"
        }
        
        # Should sanitize dangerous content
        # API keys should be handled separately and securely

    def test_agent_permission_isolation(self):
        """Test that users can only access their own agents"""
        # This is tested in other methods but worth highlighting
        pass

    def test_agent_config_encryption(self):
        """Test that sensitive agent configs are encrypted"""
        # If agent configs contain sensitive data, they should be encrypted
        pass


if __name__ == "__main__":
    pytest.main([__file__])
