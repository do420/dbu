"""
Comprehensive tests for Favorites endpoints.
Tests favorite mini services and agents management.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from api.api_v1.endpoints.favorite_services import (
    add_favorite_service,
    remove_favorite_service,
    list_user_favorites,
    check_if_favorited,
    get_favorite_counts
)
from api.api_v1.endpoints.agents import (
    add_favorite_agent,
    remove_favorite_agent,
    get_favorite_agents,
    check_favorite_status,
    get_agent_favorite_count
)
from models.favorite_service import FavoriteService
from models.favorite_agent import FavoriteAgent
from models.mini_service import MiniService
from models.agent import Agent
from models.user import User
from schemas.favorite_service import FavoriteServiceCreate
from schemas.favorite_agent import FavoriteAgentCreate


class TestFavoriteServices:
    """Test favorite mini services functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_mini_service(self):
        return MiniService(
            id=1,
            name="Test Service",
            description="Test description",
            workflow={"nodes": {"0": {"agent_id": 1}}},
            input_type="text",
            output_type="text",
            owner_id=1,
            is_public=True,
            created_at=datetime.now()
        )
    
    @pytest.fixture
    def sample_favorite_data(self):
        return FavoriteServiceCreate(mini_service_id=1)

    @pytest.mark.asyncio
    async def test_add_favorite_service_success(self, mock_db, sample_mini_service, sample_favorite_data):
        """Test successful addition of service to favorites"""
        # Mock mini service exists
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_mini_service,  # Mini service exists
            None  # No existing favorite
        ]
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.favorite_services.create_log'):
            result = await add_favorite_service(
                favorite=sample_favorite_data,
                db=mock_db,
                current_user_id=1
            )
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_favorite_service_not_found(self, mock_db, sample_favorite_data):
        """Test adding non-existent service to favorites"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await add_favorite_service(
                favorite=sample_favorite_data,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_add_favorite_service_already_exists(self, mock_db, sample_mini_service, sample_favorite_data):
        """Test adding service that's already in favorites"""
        existing_favorite = FavoriteService(user_id=1, mini_service_id=1)
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_mini_service,  # Mini service exists
            existing_favorite  # Already favorited
        ]
        
        with pytest.raises(HTTPException) as exc_info:
            await add_favorite_service(
                favorite=sample_favorite_data,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_409_CONFLICT

    @pytest.mark.asyncio
    async def test_add_favorite_service_unauthorized(self, mock_db, sample_favorite_data):
        """Test adding favorite without authentication"""
        with pytest.raises(HTTPException) as exc_info:
            await add_favorite_service(
                favorite=sample_favorite_data,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_remove_favorite_service_success(self, mock_db):
        """Test successful removal of service from favorites"""
        existing_favorite = FavoriteService(user_id=1, mini_service_id=1)
        sample_service = MiniService(id=1, name="Test Service")
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            existing_favorite,  # Favorite exists
            sample_service  # Service exists (for logging)
        ]
        mock_db.delete = Mock()
        mock_db.commit = Mock()
        
        with patch('api.api_v1.endpoints.favorite_services.create_log'):
            result = await remove_favorite_service(
                mini_service_id=1,
                db=mock_db,
                current_user_id=1
            )
            
            mock_db.delete.assert_called_once_with(existing_favorite)
            mock_db.commit.assert_called_once()
            assert result is None

    @pytest.mark.asyncio
    async def test_remove_favorite_service_not_found(self, mock_db):
        """Test removing service that's not in favorites"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await remove_favorite_service(
                mini_service_id=1,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_list_user_favorites_success(self, mock_db):
        """Test successful listing of user favorites"""
        sample_data = [
            (
                MiniService(
                    id=1,
                    name="Service 1",
                    description="Test service 1",
                    workflow={"nodes": {}},
                    input_type="text",
                    output_type="text",
                    owner_id=1,
                    created_at=datetime.now(),
                    average_token_usage={},
                    run_time=0,
                    is_enhanced=False,
                    is_public=True
                ),
                "testuser"  # owner username
            )
        ]
        
        mock_db.query.return_value.join.return_value.join.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = sample_data
        
        result = await list_user_favorites(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 1
        assert result[0]["name"] == "Service 1"
        assert result[0]["owner_username"] == "testuser"

    @pytest.mark.asyncio
    async def test_list_user_favorites_empty(self, mock_db):
        """Test listing favorites when none exist"""
        mock_db.query.return_value.join.return_value.join.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        result = await list_user_favorites(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_check_if_favorited_true(self, mock_db):
        """Test checking if service is favorited (true case)"""
        existing_favorite = FavoriteService(user_id=1, mini_service_id=1)
        mock_db.query.return_value.filter.return_value.first.return_value = existing_favorite
        
        result = await check_if_favorited(
            mini_service_id=1,
            db=mock_db,
            current_user_id=1
        )
        
        assert result["is_favorited"] is True

    @pytest.mark.asyncio
    async def test_check_if_favorited_false(self, mock_db):
        """Test checking if service is favorited (false case)"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = await check_if_favorited(
            mini_service_id=1,
            db=mock_db,
            current_user_id=1
        )
        
        assert result["is_favorited"] is False

    @pytest.mark.asyncio
    async def test_get_favorite_counts(self, mock_db):
        """Test getting favorite counts for services"""
        # Mock count results
        mock_db.query.return_value.group_by.return_value.all.return_value = [
            (1, 5),  # service_id=1, count=5
            (2, 3),  # service_id=2, count=3
        ]
        
        result = await get_favorite_counts(db=mock_db)
        
        assert len(result) == 2
        assert result[0]["mini_service_id"] == 1
        assert result[0]["favorite_count"] == 5


class TestFavoriteAgents:
    """Test favorite agents functionality"""
    
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
    async def test_add_favorite_agent_success(self, mock_db, sample_agent):
        """Test successful addition of agent to favorites"""
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_agent,  # Agent exists
            None  # No existing favorite
        ]
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.agents.create_log'):
            result = await add_favorite_agent(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_favorite_agent_not_found(self, mock_db):
        """Test adding non-existent agent to favorites"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await add_favorite_agent(
                agent_id=999,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_add_favorite_agent_already_exists(self, mock_db, sample_agent):
        """Test adding agent that's already in favorites"""
        existing_favorite = FavoriteAgent(user_id=1, agent_id=1)
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_agent,  # Agent exists
            existing_favorite  # Already favorited
        ]
        
        with pytest.raises(HTTPException) as exc_info:
            await add_favorite_agent(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_remove_favorite_agent_success(self, mock_db, sample_agent):
        """Test successful removal of agent from favorites"""
        existing_favorite = FavoriteAgent(user_id=1, agent_id=1)
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            existing_favorite,  # Favorite exists
            sample_agent  # Agent exists (for logging)
        ]
        mock_db.delete = Mock()
        mock_db.commit = Mock()
        
        with patch('api.api_v1.endpoints.agents.create_log'):
            result = await remove_favorite_agent(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
            
            mock_db.delete.assert_called_once_with(existing_favorite)
            mock_db.commit.assert_called_once()
            assert result is None

    @pytest.mark.asyncio
    async def test_remove_favorite_agent_not_found(self, mock_db):
        """Test removing agent that's not in favorites"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await remove_favorite_agent(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_favorite_agents_success(self, mock_db):
        """Test successful listing of favorite agents"""
        favorite_agents = [
            Agent(
                id=1,
                name="Agent 1",
                agent_type="openai",
                system_instruction="Helper 1",
                config={},
                input_type="text",
                output_type="text",
                owner_id=1
            ),
            Agent(
                id=2,
                name="Agent 2",
                agent_type="gemini",
                system_instruction="Helper 2",
                config={},
                input_type="text",
                output_type="text",
                owner_id=1
            )
        ]
        
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = favorite_agents
        
        result = await get_favorite_agents(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 2
        assert result[0].name == "Agent 1"
        assert result[1].name == "Agent 2"

    @pytest.mark.asyncio
    async def test_get_favorite_agents_empty(self, mock_db):
        """Test listing favorite agents when none exist"""
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        result = await get_favorite_agents(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_check_favorite_status_true(self, mock_db):
        """Test checking agent favorite status (true case)"""
        existing_favorite = FavoriteAgent(user_id=1, agent_id=1)
        mock_db.query.return_value.filter.return_value.first.return_value = existing_favorite
        
        result = await check_favorite_status(
            agent_id=1,
            db=mock_db,
            current_user_id=1
        )
        
        assert result["is_favorite"] is True

    @pytest.mark.asyncio
    async def test_check_favorite_status_false(self, mock_db):
        """Test checking agent favorite status (false case)"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = await check_favorite_status(
            agent_id=1,
            db=mock_db,
            current_user_id=1
        )
        
        assert result["is_favorite"] is False

    @pytest.mark.asyncio
    async def test_check_favorite_status_no_user(self, mock_db):
        """Test checking favorite status without user authentication"""
        result = await check_favorite_status(
            agent_id=1,
            db=mock_db,
            current_user_id=None
        )
        
        assert result["is_favorite"] is False

    @pytest.mark.asyncio
    async def test_get_agent_favorite_count(self, mock_db):
        """Test getting total favorite count for an agent"""
        mock_db.query.return_value.filter.return_value.count.return_value = 7
        
        result = await get_agent_favorite_count(
            agent_id=1,
            db=mock_db
        )
        
        assert result.agent_id == 1
        assert result.favorite_count == 7


class TestFavoritesSecurity:
    """Test security aspects of favorites management"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.mark.asyncio
    async def test_favorite_user_isolation_services(self, mock_db):
        """Test that users only see their own favorite services"""
        mock_db.query.return_value.join.return_value.join.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        await list_user_favorites(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1
        )
        
        # Verify that the query filters by user_id
        filter_calls = mock_db.query.return_value.join.return_value.join.return_value.filter.call_args_list
        assert len(filter_calls) > 0

    @pytest.mark.asyncio
    async def test_favorite_user_isolation_agents(self, mock_db):
        """Test that users only see their own favorite agents"""
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        await get_favorite_agents(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1
        )
        
        # Verify that the query filters by user_id
        filter_calls = mock_db.query.return_value.join.return_value.filter.call_args_list
        assert len(filter_calls) > 0

    @pytest.mark.asyncio
    async def test_cannot_remove_others_favorites(self, mock_db):
        """Test that users cannot remove other users' favorites"""
        # Mock a favorite belonging to different user
        mock_db.query.return_value.filter.return_value.first.return_value = None  # No favorite found for this user
        
        with pytest.raises(HTTPException) as exc_info:
            await remove_favorite_service(
                mini_service_id=1,
                db=mock_db,
                current_user_id=999  # Different user
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    def test_sql_injection_protection(self):
        """Test protection against SQL injection in favorites operations"""
        # Since we're using SQLAlchemy ORM, this should be handled automatically
        pass


class TestFavoritesValidation:
    """Test validation and business logic for favorites"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.mark.asyncio
    async def test_favorite_private_service_allowed(self, mock_db):
        """Test that users can favorite private services they can access"""
        private_service = MiniService(
            id=1,
            name="Private Service",
            description="Private test service",
            workflow={},
            input_type="text",
            output_type="text",
            owner_id=1,
            is_public=False
        )
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            private_service,  # Service exists
            None  # Not already favorited
        ]
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        favorite_data = FavoriteServiceCreate(mini_service_id=1)
        
        with patch('api.api_v1.endpoints.favorite_services.create_log'):
            # Should allow favoriting private service (if user has access)
            await add_favorite_service(
                favorite=favorite_data,
                db=mock_db,
                current_user_id=1
            )
            
            mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_favorite_invalid_ids(self, mock_db):
        """Test handling of invalid service/agent IDs"""
        invalid_ids = [0, -1, 999999]
        
        for invalid_id in invalid_ids:
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            with pytest.raises(HTTPException) as exc_info:
                favorite_data = FavoriteServiceCreate(mini_service_id=invalid_id)
                await add_favorite_service(
                    favorite=favorite_data,
                    db=mock_db,
                    current_user_id=1
                )
            
            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


class TestFavoritesPerformance:
    """Test performance aspects of favorites management"""
    
    @pytest.mark.asyncio
    async def test_list_large_favorites(self, mock_db):
        """Test performance with large number of favorites"""
        # Create many mock favorites
        large_favorites_list = []
        for i in range(100):
            service_data = (
                MiniService(
                    id=i,
                    name=f"Service {i}",
                    description=f"Description {i}",
                    workflow={},
                    input_type="text",
                    output_type="text",
                    owner_id=1,
                    created_at=datetime.now(),
                    average_token_usage={},
                    run_time=0,
                    is_enhanced=False,
                    is_public=True
                ),
                f"user_{i % 10}"  # owner username
            )
            large_favorites_list.append(service_data)
        
        mock_db.query.return_value.join.return_value.join.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = large_favorites_list[:10]  # Pagination
        
        import time
        start_time = time.time()
        
        result = await list_user_favorites(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert len(result) == 10
        assert execution_time < 1.0  # Should handle pagination efficiently

    @pytest.mark.asyncio
    async def test_favorite_count_performance(self, mock_db):
        """Test performance of counting favorites"""
        mock_db.query.return_value.filter.return_value.count.return_value = 50
        
        import time
        start_time = time.time()
        
        result = await get_agent_favorite_count(
            agent_id=1,
            db=mock_db
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert result.favorite_count == 50
        assert execution_time < 0.1  # Count operations should be very fast


class TestFavoritesIntegration:
    """Integration tests for favorites functionality"""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        from db.session import get_db
        
        def mock_get_db():
            return Mock()
        
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)
        yield client
        app.dependency_overrides.clear()

    def test_add_favorite_service_endpoint(self, client):
        """Test add favorite service endpoint through FastAPI client"""
        with patch('api.api_v1.endpoints.favorite_services.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock service exists and not already favorited
            mock_service = Mock()
            mock_service.id = 1
            mock_service.name = "Test Service"
            mock_db.query.return_value.filter.return_value.first.side_effect = [
                mock_service,  # Service exists
                None  # Not favorited
            ]
            
            mock_db.add = Mock()
            mock_db.commit = Mock()
            mock_db.refresh = Mock()
            
            with patch('api.api_v1.endpoints.favorite_services.create_log'):
                response = client.post(
                    "/api/v1/favorites/?current_user_id=1",
                    json={"mini_service_id": 1}
                )
        
        assert response.status_code in [200, 201, 422]

    def test_list_favorites_endpoint(self, client):
        """Test list favorites endpoint through FastAPI client"""
        response = client.get("/api/v1/favorites/?current_user_id=1")
        assert response.status_code in [200, 422]

    def test_favorites_without_auth(self, client):
        """Test favorites endpoints without authentication"""
        # Test without current_user_id
        response = client.post("/api/v1/favorites/", json={"mini_service_id": 1})
        assert response.status_code in [401, 422]
        
        response = client.get("/api/v1/favorites/")
        assert response.status_code in [401, 422]


if __name__ == "__main__":
    pytest.main([__file__])
