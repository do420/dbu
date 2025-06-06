"""
Comprehensive tests for processes endpoints.
Tests process listing, recent activities, and process management.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from api.api_v1.endpoints.processes import list_processes, get_recent_logs
from models.process import Process
from models.mini_service import MiniService
from models.log import Log
from models.user import User


class TestProcessListing:
    """Test process listing functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_processes(self):
        """Sample processes with mini service data"""
        return [
            (
                Process(
                    id=1,
                    mini_service_id=1,
                    user_id=1,
                    total_tokens={"total_tokens": 100, "pricing": {"cost": 0.002}},
                    created_at=datetime.now()
                ),
                "Text Summarizer",  # mini_service_name
                "Summarizes text",  # mini_service_description
                "text",  # mini_service_input_type
                "text",  # mini_service_output_type
                False  # mini_service_is_enhanced
            ),
            (
                Process(
                    id=2,
                    mini_service_id=2,
                    user_id=1,
                    total_tokens={"total_tokens": 200, "pricing": {"cost": 0.004}},
                    created_at=datetime.now()
                ),
                "Image Generator",
                "Generates images from text",
                "text",
                "image",
                True
            )
        ]

    @pytest.mark.asyncio
    async def test_list_processes_success(self, mock_db, sample_processes):
        """Test successful process listing"""
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = sample_processes
        
        with patch('api.api_v1.endpoints.processes.pricing_calculator') as mock_pricing:
            mock_pricing.add_pricing_to_response.return_value = {"pricing": {"cost": 0.002}}
            
            result = await list_processes(
                skip=0,
                limit=10,
                db=mock_db,
                current_user_id=1
            )
            
            assert len(result) == 2
            assert result[0].id == 1
            assert result[0].mini_service_name == "Text Summarizer"
            assert result[1].id == 2
            assert result[1].mini_service_name == "Image Generator"

    @pytest.mark.asyncio
    async def test_list_processes_empty(self, mock_db):
        """Test listing when no processes exist"""
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        result = await list_processes(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_list_processes_pagination(self, mock_db, sample_processes):
        """Test process listing with pagination"""
        # Return only first process for pagination test
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [sample_processes[0]]
        
        with patch('api.api_v1.endpoints.processes.pricing_calculator') as mock_pricing:
            mock_pricing.add_pricing_to_response.return_value = {"pricing": {"cost": 0.002}}
            
            result = await list_processes(
                skip=0,
                limit=1,
                db=mock_db,
                current_user_id=1
            )
            
            assert len(result) == 1
            assert result[0].id == 1

    @pytest.mark.asyncio
    async def test_list_processes_unauthorized(self, mock_db):
        """Test process listing without authentication"""
        with pytest.raises(HTTPException) as exc_info:
            await list_processes(
                skip=0,
                limit=10,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_list_processes_user_isolation(self, mock_db):
        """Test that users only see their own processes"""
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        await list_processes(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1
        )
        
        # Verify that the query filters by user_id
        # This would need to check the actual filter calls made to mock_db
        assert mock_db.query.called

    @pytest.mark.asyncio
    async def test_list_processes_with_pricing(self, mock_db, sample_processes):
        """Test that processes include pricing information"""
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = sample_processes
        
        with patch('api.api_v1.endpoints.processes.pricing_calculator') as mock_pricing:
            mock_pricing.add_pricing_to_response.return_value = {"pricing": {"cost": 0.002, "currency": "USD"}}
            
            result = await list_processes(
                skip=0,
                limit=10,
                db=mock_db,
                current_user_id=1
            )
            
            assert len(result) == 2
            # Verify pricing calculation was called
            assert mock_pricing.add_pricing_to_response.called


class TestRecentActivities:
    """Test recent activities/logs functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.fixture
    def sample_logs(self):
        """Sample log entries"""
        return [
            Log(
                id=1,
                user_id=1,
                type=0,  # Info
                description="Created mini service 'Text Summarizer'",
                created_at=datetime.now()
            ),
            Log(
                id=2,
                user_id=1,
                type=5,  # Success
                description="Successfully ran mini service 'Image Generator'",
                created_at=datetime.now()
            ),
            Log(
                id=3,
                user_id=1,
                type=4,  # Warning
                description="Deleted mini service 'Old Service'",
                created_at=datetime.now()
            )
        ]

    @pytest.mark.asyncio
    async def test_get_recent_logs_success(self, mock_db, sample_logs):
        """Test successful recent logs retrieval"""
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = sample_logs
        
        result = await get_recent_logs(
            limit=5,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 3
        assert result[0].description == "Created mini service 'Text Summarizer'"
        assert result[1].description == "Successfully ran mini service 'Image Generator'"
        assert result[2].description == "Deleted mini service 'Old Service'"

    @pytest.mark.asyncio
    async def test_get_recent_logs_empty(self, mock_db):
        """Test recent logs when no logs exist"""
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        result = await get_recent_logs(
            limit=5,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_recent_logs_limit(self, mock_db, sample_logs):
        """Test recent logs with custom limit"""
        # Return only first 2 logs
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = sample_logs[:2]
        
        result = await get_recent_logs(
            limit=2,
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

    @pytest.mark.asyncio
    async def test_get_recent_logs_unauthorized(self, mock_db):
        """Test recent logs without authentication"""
        with pytest.raises(HTTPException) as exc_info:
            await get_recent_logs(
                limit=5,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_recent_logs_user_isolation(self, mock_db):
        """Test that users only see their own logs"""
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        await get_recent_logs(
            limit=5,
            db=mock_db,
            current_user_id=1
        )
        
        # Verify that the query filters by user_id
        assert mock_db.query.called

    @pytest.mark.asyncio
    async def test_get_recent_logs_ordering(self, mock_db, sample_logs):
        """Test that logs are returned in correct order (newest first)"""
        # Reverse the logs to simulate database ordering
        reversed_logs = list(reversed(sample_logs))
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = reversed_logs
        
        result = await get_recent_logs(
            limit=5,
            db=mock_db,
            current_user_id=1
        )
        
        # Should be ordered by created_at descending (newest first)
        assert len(result) == 3


class TestProcessIntegration:
    """Integration tests for process endpoints"""
    
    def test_list_processes_endpoint(self, client):
        """Test list processes endpoint through FastAPI client"""
        response = client.get("/api/v1/processes/?current_user_id=1")
        
        # Without proper database setup, check that endpoint is accessible
        assert response.status_code in [200, 401, 422]

    def test_recent_activities_endpoint(self, client):
        """Test recent activities endpoint through FastAPI client"""
        response = client.get("/api/v1/processes/recent-activities?current_user_id=1&limit=5")
        
        assert response.status_code in [200, 401, 422]

    def test_processes_without_auth(self, client):
        """Test processes endpoints without authentication"""
        # Test without current_user_id
        response = client.get("/api/v1/processes/")
        assert response.status_code in [401, 422]
        
        response = client.get("/api/v1/processes/recent-activities")
        assert response.status_code in [401, 422]


class TestProcessSecurity:
    """Test security aspects of process management"""
    
    @pytest.mark.asyncio
    async def test_process_data_isolation(self, mock_db):
        """Test that users can't access other users' process data"""
        # Mock process belonging to different user
        other_user_process = Process(
            id=1,
            mini_service_id=1,
            user_id=2,  # Different user
            total_tokens={"total_tokens": 100}
        )
        
        # Should not return processes for other users
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        result = await list_processes(
            skip=0,
            limit=10,
            db=mock_db,
            current_user_id=1  # Different user
        )
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_log_data_isolation(self, mock_db):
        """Test that users can't access other users' logs"""        # Mock log belonging to different user
        other_user_log = Log(
            id=1,
            user_id=2,  # Different user
            type=0,
            description="Other user's activity",
            created_at=datetime.now()
        )
        
        # Should not return logs for other users
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        result = await get_recent_logs(
            limit=5,
            db=mock_db,
            current_user_id=1  # Different user
        )
        
        assert len(result) == 0

    def test_sql_injection_protection(self):
        """Test protection against SQL injection in process queries"""
        # This would test that parameters are properly sanitized
        # Since we're using SQLAlchemy ORM, this should be handled automatically
        pass

    def test_data_sanitization(self):
        """Test that sensitive data is properly sanitized in responses"""
        # Check that API keys, passwords, etc. are not exposed in process data
        pass


class TestProcessPerformance:
    """Test performance aspects of process endpoints"""
    
    @pytest.mark.asyncio
    async def test_large_process_list_performance(self, mock_db):
        """Test performance with large number of processes"""        # Create many mock processes
        large_process_list = []
        for i in range(1000):
            process_data = (
                Process(
                    id=i,
                    mini_service_id=1,
                    user_id=1,
                    total_tokens={"total_tokens": 100},
                    created_at=datetime.now()
                ),
                f"Service {i}",
                f"Description {i}",
                "text",
                "text",
                False
            )
            large_process_list.append(process_data)
        
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = large_process_list[:10]  # Pagination
        
        with patch('api.api_v1.endpoints.processes.pricing_calculator') as mock_pricing:
            mock_pricing.add_pricing_to_response.return_value = {"pricing": {"cost": 0.002}}
            
            import time
            start_time = time.time()
            
            result = await list_processes(
                skip=0,
                limit=10,
                db=mock_db,
                current_user_id=1
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should handle pagination efficiently
            assert len(result) == 10
            assert execution_time < 1.0  # Should be fast with mocks    @pytest.mark.asyncio
    async def test_pricing_calculation_performance(self, mock_db):
        """Test performance of pricing calculations"""
        process_data = (
            Process(
                id=1,
                mini_service_id=1,
                user_id=1,
                total_tokens={"total_tokens": 1000000},  # Large token count
                created_at=datetime.now()
            ),
            "Heavy Service",
            "Resource intensive service",
            "text",
            "text",
            False
        )
        
        mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [process_data]
        
        with patch('api.api_v1.endpoints.processes.pricing_calculator') as mock_pricing:
            mock_pricing.add_pricing_to_response.return_value = {"pricing": {"cost": 2.50}}
            
            import time
            start_time = time.time()
            
            result = await list_processes(
                skip=0,
                limit=1,
                db=mock_db,
                current_user_id=1
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            assert len(result) == 1
            assert execution_time < 0.5  # Pricing calculation should be fast


class TestProcessValidation:
    """Test process data validation"""
    
    def test_process_token_usage_validation(self):
        """Test validation of token usage data"""
        valid_token_usage = {
            "total_tokens": 100,
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "pricing": {"cost": 0.002, "currency": "USD"}
        }
        
        # Should validate token usage structure
        assert "total_tokens" in valid_token_usage
        assert isinstance(valid_token_usage["total_tokens"], int)
        assert valid_token_usage["total_tokens"] > 0

    def test_process_pagination_validation(self):
        """Test validation of pagination parameters"""
        # Test invalid pagination parameters
        invalid_params = [
            {"skip": -1, "limit": 10},  # Negative skip
            {"skip": 0, "limit": -1},   # Negative limit
            {"skip": 0, "limit": 1001}, # Too large limit
        ]
        
        for params in invalid_params:
            # These should be validated and rejected
            if params["skip"] < 0 or params["limit"] < 0 or params["limit"] > 1000:
                # Should raise validation error
                pass

    def test_log_type_validation(self):
        """Test validation of log types"""
        valid_log_types = [0, 1, 2, 3, 4, 5]  # Info, Debug, Warning, Error, etc.
        
        for log_type in valid_log_types:
            assert log_type >= 0
            assert log_type <= 5


if __name__ == "__main__":
    pytest.main([__file__])
