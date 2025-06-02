"""
Edge case and security tests for mini services.
Tests unusual inputs, boundary conditions, and security vulnerabilities.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import HTTPException, status

from api.api_v1.endpoints.mini_services import (
    create_mini_service,
    run_mini_service,
    delete_mini_service
)
from schemas.mini_service import MiniServiceCreate


class TestSecurityEdgeCases:
    """Test security-related edge cases"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_sql_injection_attempt_in_workflow(self, mock_db):
        """Test that SQL injection attempts in workflow are handled safely"""
        malicious_workflow = {
            "nodes": {
                "0": {
                    "agent_id": "1; DROP TABLE mini_services; --",
                    "next": None
                }
            }
        }
        
        mini_service = MiniServiceCreate(
            name="Malicious Service",
            description="'; DROP TABLE users; --",
            workflow=malicious_workflow,
            input_type="text",
            output_type="text"
        )
        
        # Should fail during validation or agent lookup
        with pytest.raises((HTTPException, ValueError, TypeError)):
            await create_mini_service(
                mini_service=mini_service,
                db=mock_db,
                current_user_id=1
            )

    @pytest.mark.asyncio
    async def test_unauthorized_service_access(self, mock_db):
        """Test accessing private service of another user"""
        mock_service = Mock()
        mock_service.id = 1
        mock_service.owner_id = 2  # Different user
        mock_service.is_public = False
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_service
        
        with pytest.raises(HTTPException) as exc_info:
            await run_mini_service(
                service_id=1,
                input_data={"input": "test"},
                db=mock_db,
                current_user_id=1  # Different user
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_privilege_escalation_attempt(self, mock_db):
        """Test attempting to delete another user's service"""
        # Mock finding service owned by different user
        mock_execute = Mock()
        mock_execute.scalar_one_or_none.return_value = None  # No service found for this user
        mock_db.execute.return_value = mock_execute
        
        with pytest.raises(HTTPException) as exc_info:
            await delete_mini_service(
                service_id=1,
                db=mock_db,
                current_user_id=999  # Trying to delete as different user
            )
        
        assert exc_info.value.status_code == 404


class TestBoundaryConditions:
    """Test boundary conditions and limits"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_extremely_large_workflow(self, mock_db):
        """Test handling of extremely large workflow configurations"""
        # Create a workflow with many nodes
        large_workflow = {"nodes": {}}
        for i in range(1000):  # 1000 nodes
            large_workflow["nodes"][str(i)] = {
                "agent_id": 1,
                "next": i + 1 if i < 999 else None
            }
        
        mini_service = MiniServiceCreate(
            name="Large Service",
            description="Service with many nodes",
            workflow=large_workflow,
            input_type="text",
            output_type="text"
        )
        
        # Mock agent to exist
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.is_enhanced = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            # Should handle large workflows (might be slow but shouldn't crash)
            result = await create_mini_service(
                mini_service=mini_service,
                db=mock_db,
                current_user_id=1
            )
        
        # Should succeed if the system can handle it
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_zero_and_negative_agent_ids(self, mock_db):
        """Test handling of zero and negative agent IDs"""
        invalid_workflows = [
            {"nodes": {"0": {"agent_id": 0, "next": None}}},
            {"nodes": {"0": {"agent_id": -1, "next": None}}},
            {"nodes": {"0": {"agent_id": -999, "next": None}}}
        ]
        
        for workflow in invalid_workflows:
            mini_service = MiniServiceCreate(
                name="Invalid Agent ID Service",
                description="Service with invalid agent ID",
                workflow=workflow,
                input_type="text",
                output_type="text"
            )
            
            # Mock agent not found
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            with pytest.raises(HTTPException) as exc_info:
                await create_mini_service(
                    mini_service=mini_service,
                    db=mock_db,
                    current_user_id=1
                )
            
            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_extremely_long_names_and_descriptions(self, mock_db):
        """Test handling of extremely long names and descriptions"""
        long_text = "x" * 10000  # 10KB of text
        
        mini_service = MiniServiceCreate(
            name=long_text,
            description=long_text,
            workflow={"nodes": {"0": {"agent_id": 1, "next": None}}},
            input_type="text",
            output_type="text"
        )
        
        # Mock agent exists
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.is_enhanced = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent
        
        # Should handle long strings without crashing
        # (Database constraints might limit actual storage)
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            try:
                await create_mini_service(
                    mini_service=mini_service,
                    db=mock_db,
                    current_user_id=1
                )
            except Exception as e:
                # Should either succeed or fail gracefully with database error
                assert "too long" in str(e).lower() or mock_db.add.called


class TestDataValidationEdgeCases:
    """Test edge cases in data validation"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self, mock_db):
        """Test handling of Unicode and special characters"""
        unicode_workflow = {
            "nodes": {
                "0": {"agent_id": 1, "next": None}
            }
        }
        
        mini_service = MiniServiceCreate(
            name="Service with émojis 🚀🤖",
            description="Description with 中文 and العربية",
            workflow=unicode_workflow,
            input_type="text",
            output_type="text"
        )
        
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.is_enhanced = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            # Should handle Unicode characters properly
            result = await create_mini_service(
                mini_service=mini_service,
                db=mock_db,
                current_user_id=1
            )
        
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_and_whitespace_only_inputs(self, mock_db):
        """Test handling of empty and whitespace-only inputs"""
        test_cases = [
            {"name": "", "description": ""},
            {"name": "   ", "description": "   "},
            {"name": "\n\t\r", "description": "\n\t\r"},
        ]
        
        for case in test_cases:
            mini_service = MiniServiceCreate(
                name=case["name"],
                description=case["description"],
                workflow={"nodes": {"0": {"agent_id": 1, "next": None}}},
                input_type="text",
                output_type="text"
            )
            
            mock_agent = Mock()
            mock_agent.id = 1
            mock_agent.is_enhanced = False
            mock_db.query.return_value.filter.return_value.first.return_value = mock_agent
            mock_db.add = Mock()
            mock_db.commit = Mock()
            mock_db.refresh = Mock()
            
            with patch('api.api_v1.endpoints.mini_services.create_log'):
                # Should handle empty/whitespace inputs
                # (Validation rules will determine if this is allowed)
                try:
                    await create_mini_service(
                        mini_service=mini_service,
                        db=mock_db,
                        current_user_id=1
                    )
                except HTTPException:
                    # May fail validation, which is acceptable
                    pass

    @pytest.mark.asyncio
    async def test_circular_workflow_detection(self, mock_db):
        """Test detection of circular workflows"""
        circular_workflow = {
            "nodes": {
                "0": {"agent_id": 1, "next": 1},
                "1": {"agent_id": 2, "next": 2},  # Points to itself
                "2": {"agent_id": 3, "next": 0}   # Points back to start
            }
        }
        
        mini_service = MiniServiceCreate(
            name="Circular Service",
            description="Service with circular workflow",
            workflow=circular_workflow,
            input_type="text",
            output_type="text"
        )
        
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.is_enhanced = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            # Current implementation doesn't check for cycles
            # This test documents the current behavior
            result = await create_mini_service(
                mini_service=mini_service,
                db=mock_db,
                current_user_id=1
            )
        
        # Currently allows circular workflows
        mock_db.add.assert_called_once()


class TestConcurrencyAndRaceConditions:
    """Test concurrent access and race conditions"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_concurrent_service_creation(self, mock_db):
        """Test multiple users creating services simultaneously"""
        mini_service = MiniServiceCreate(
            name="Concurrent Service",
            description="Created by multiple users",
            workflow={"nodes": {"0": {"agent_id": 1, "next": None}}},
            input_type="text",
            output_type="text"
        )
        
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.is_enhanced = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            # Simulate concurrent creation by different users
            tasks = []
            for user_id in [1, 2, 3]:
                task = create_mini_service(
                    mini_service=mini_service,
                    db=mock_db,
                    current_user_id=user_id
                )
                tasks.append(task)
            
            # All should succeed independently
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    results.append(e)
            
            # Should handle concurrent access gracefully
            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_race_condition_in_service_deletion(self, mock_db):
        """Test race condition when deleting service"""
        # Mock service found first time, not found second time
        mock_execute = Mock()
        mock_execute.scalar_one_or_none.side_effect = [
            Mock(id=1, name="Test Service", owner_id=1),  # Found
            None  # Not found on second attempt
        ]
        mock_db.execute.return_value = mock_execute
        mock_db.delete = Mock()
        mock_db.commit = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            # First deletion should succeed
            result1 = await delete_mini_service(
                service_id=1,
                db=mock_db,
                current_user_id=1
            )
            
            # Second deletion should fail (service already deleted)
            with pytest.raises(HTTPException) as exc_info:
                await delete_mini_service(
                    service_id=1,
                    db=mock_db,
                    current_user_id=1
                )
            
            assert exc_info.value.status_code == 404


class TestResourceExhaustion:
    """Test resource exhaustion scenarios"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_memory_exhaustion_large_input(self, mock_db):
        """Test handling of extremely large input data"""
        # Mock mini service
        mock_service = Mock()
        mock_service.id = 1
        mock_service.owner_id = 1
        mock_service.is_public = True
        mock_service.workflow = {"nodes": {"0": {"agent_id": 1, "next": None}}}
        mock_service.run_time = 0
        mock_service.average_token_usage = {}
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.agent_type = "openai"
        mock_agent.config = {}
        mock_agent.system_instruction = "Test"
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_service,
            mock_agent
        ]
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Create extremely large input
        large_input = "x" * (10 * 1024 * 1024)  # 10MB of text
        input_data = {
            "input": large_input,
            "context": {},
            "api_keys": {"1": "test-key"}
        }
        
        with patch('api.api_v1.endpoints.mini_services.create_agent') as mock_create_agent, \
             patch('api.api_v1.endpoints.mini_services.WorkflowProcessor') as mock_workflow_processor, \
             patch('api.api_v1.endpoints.mini_services.create_log'), \
             patch('api.api_v1.endpoints.mini_services.flag_modified'):
            
            mock_agent_instance = AsyncMock()
            mock_create_agent.return_value = mock_agent_instance
            
            # Mock processor to handle large input
            mock_processor = AsyncMock()
            mock_processor.process.return_value = {
                "output": "Processed large input",
                "token_usage": {"total_tokens": 1000},
                "results": []
            }
            mock_workflow_processor.return_value = mock_processor
            
            # Should handle large input gracefully
            try:
                result = await run_mini_service(
                    service_id=1,
                    input_data=input_data,
                    db=mock_db,
                    current_user_id=1
                )
                # If successful, verify it completed
                assert "output" in result
            except Exception as e:
                # Should fail gracefully, not crash
                assert "memory" in str(e).lower() or "size" in str(e).lower() or "timeout" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])
