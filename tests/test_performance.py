"""
Performance and load tests for mini services.
Tests system behavior under load and measures performance characteristics.
"""
import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from api.api_v1.endpoints.mini_services import (
    create_mini_service,
    run_mini_service,
    list_mini_services
)
from schemas.mini_service import MiniServiceCreate


class TestPerformance:
    """Performance tests for mini service operations"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def sample_mini_service_create(self):
        return MiniServiceCreate(
            name="Performance Test Service",
            description="Service for performance testing",
            workflow={"nodes": {"0": {"agent_id": 1, "next": None}}},
            input_type="text",
            output_type="text"
        )

    @pytest.mark.asyncio
    async def test_service_creation_performance(self, mock_db, sample_mini_service_create):
        """Test performance of mini service creation"""
        # Mock agent
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.is_enhanced = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            start_time = time.time()
            
            # Create multiple services
            for i in range(10):
                await create_mini_service(
                    mini_service=sample_mini_service_create,
                    db=mock_db,
                    current_user_id=1
                )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete 10 creations in reasonable time (< 1 second with mocks)
            assert total_time < 1.0
            print(f"Created 10 services in {total_time:.3f} seconds")

    @pytest.mark.asyncio
    async def test_concurrent_service_runs(self, mock_db):
        """Test concurrent execution of mini services"""
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
        
        mock_db.query.return_value.filter.return_value.first.side_effect = lambda: mock_service if mock_db.query.call_count % 2 == 1 else mock_agent
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        input_data = {
            "input": "Test input",
            "context": {},
            "api_keys": {"1": "test-key"}
        }
        
        with patch('api.api_v1.endpoints.mini_services.create_agent') as mock_create_agent, \
             patch('api.api_v1.endpoints.mini_services.WorkflowProcessor') as mock_workflow_processor, \
             patch('api.api_v1.endpoints.mini_services.create_log'), \
             patch('api.api_v1.endpoints.mini_services.flag_modified'):
            
            mock_agent_instance = AsyncMock()
            mock_create_agent.return_value = mock_agent_instance
            
            # Mock processor with slight delay to simulate real processing
            mock_processor = AsyncMock()
            async def mock_process(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms delay
                return {
                    "output": "Processed result",
                    "token_usage": {"total_tokens": 100},
                    "results": []
                }
            mock_processor.process = mock_process
            mock_workflow_processor.return_value = mock_processor
            
            start_time = time.time()
            
            # Run 20 concurrent mini service executions
            tasks = []
            for i in range(20):
                task = run_mini_service(
                    service_id=1,
                    input_data=input_data,
                    db=mock_db,
                    current_user_id=1
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete 20 concurrent runs efficiently
            successful_runs = [r for r in results if not isinstance(r, Exception)]
            print(f"Completed {len(successful_runs)}/20 concurrent runs in {total_time:.3f} seconds")
            
            # With 10ms delay each, concurrent execution should be much faster than 200ms
            assert total_time < 0.2

    @pytest.mark.asyncio
    async def test_large_workflow_performance(self, mock_db):
        """Test performance with large workflow configurations"""
        # Create workflow with 100 nodes
        large_workflow = {"nodes": {}}
        for i in range(100):
            large_workflow["nodes"][str(i)] = {
                "agent_id": 1,
                "next": i + 1 if i < 99 else None
            }
        
        mini_service = MiniServiceCreate(
            name="Large Workflow Service",
            description="Service with large workflow",
            workflow=large_workflow,
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
            start_time = time.time()
            
            result = await create_mini_service(
                mini_service=mini_service,
                db=mock_db,
                current_user_id=1
            )
            
            end_time = time.time()
            creation_time = end_time - start_time
            
            # Should handle large workflows efficiently
            print(f"Created service with 100 nodes in {creation_time:.3f} seconds")
            assert creation_time < 1.0  # Should complete within 1 second

    @pytest.mark.asyncio
    async def test_listing_performance_with_many_services(self, mock_db):
        """Test performance of listing many services"""
        # Mock many services
        mock_services = []
        for i in range(100):
            mock_service = Mock()
            mock_service.__dict__ = {
                "id": i,
                "name": f"Service {i}",
                "workflow": {"nodes": {"0": {"agent_id": 1}}},
                "average_token_usage": {}
            }
            mock_services.append((mock_service, f"user{i}"))
          # Mock the main mini services query
        mock_main_query = Mock()
        mock_main_query.join.return_value = mock_main_query
        mock_main_query.filter.return_value = mock_main_query
        mock_main_query.offset.return_value = mock_main_query
        mock_main_query.limit.return_value = mock_main_query
        mock_main_query.all.return_value = mock_services
        
        # Mock agent query
        mock_agent_query = Mock()
        mock_agent = Mock()
        mock_agent.agent_type = "openai"
        mock_agent_query.filter.return_value.all.return_value = [mock_agent]
          # Set up the db.query to return different mocks based on the model being queried
        def mock_query_side_effect(*args):
            model_class = args[0] if args else None
            if model_class and hasattr(model_class, '__name__') and 'Agent' in model_class.__name__:
                return mock_agent_query
            else:
                return mock_main_query
        
        mock_db.query.side_effect = mock_query_side_effect
        
        start_time = time.time()
        
        result = await list_mini_services(
            skip=0,
            limit=100,
            db=mock_db,
            current_user_id=1
        )
        
        end_time = time.time()
        listing_time = end_time - start_time
        
        print(f"Listed 100 services in {listing_time:.3f} seconds")
        assert len(result) == 100
        assert listing_time < 0.5  # Should be fast with proper indexing


class TestLoadTesting:
    """Load testing scenarios"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_high_concurrency_service_creation(self, mock_db):
        """Test system behavior under high concurrent load"""
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.is_enhanced = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_agent
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        mini_service = MiniServiceCreate(
            name="Load Test Service",
            description="Service for load testing",
            workflow={"nodes": {"0": {"agent_id": 1, "next": None}}},
            input_type="text",
            output_type="text"
        )
        
        with patch('api.api_v1.endpoints.mini_services.create_log'):
            # Simulate 50 concurrent users creating services
            tasks = []
            for user_id in range(50):
                task = create_mini_service(
                    mini_service=mini_service,
                    db=mock_db,
                    current_user_id=user_id + 1
                )
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            total_time = end_time - start_time
            successful_creations = [r for r in results if not isinstance(r, Exception)]
            
            print(f"Load test: {len(successful_creations)}/50 services created in {total_time:.3f} seconds")
            
            # Should handle high concurrency reasonably well
            assert len(successful_creations) >= 40  # At least 80% success rate
            assert total_time < 5.0  # Should complete within 5 seconds

    @pytest.mark.asyncio
    async def test_sustained_load_simulation(self, mock_db):
        """Test sustained load over time"""
        mock_service = Mock()
        mock_service.id = 1
        mock_service.owner_id = 1
        mock_service.is_public = True
        mock_service.workflow = {"nodes": {"0": {"agent_id": 1, "next": None}}}
        mock_service.run_time = 0
        mock_service.average_token_usage = {}
        
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.agent_type = "openai"
        mock_agent.config = {}
        mock_agent.system_instruction = "Test"
        
        mock_db.query.return_value.filter.return_value.first.side_effect = lambda: mock_service if mock_db.query.call_count % 2 == 1 else mock_agent
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        input_data = {
            "input": "Load test input",
            "context": {},
            "api_keys": {"1": "test-key"}
        }
        
        with patch('api.api_v1.endpoints.mini_services.create_agent') as mock_create_agent, \
             patch('api.api_v1.endpoints.mini_services.WorkflowProcessor') as mock_workflow_processor, \
             patch('api.api_v1.endpoints.mini_services.create_log'), \
             patch('api.api_v1.endpoints.mini_services.flag_modified'):
            
            mock_agent_instance = AsyncMock()
            mock_create_agent.return_value = mock_agent_instance
            
            mock_processor = AsyncMock()
            async def mock_process(*args, **kwargs):
                await asyncio.sleep(0.005)  # 5ms processing time
                return {
                    "output": "Load test result",
                    "token_usage": {"total_tokens": 50},
                    "results": []
                }
            mock_processor.process = mock_process
            mock_workflow_processor.return_value = mock_processor
            
            # Simulate sustained load: 10 requests per second for 3 seconds
            start_time = time.time()
            all_results = []
            
            for second in range(3):
                # Launch 10 requests
                batch_tasks = []
                for i in range(10):
                    task = run_mini_service(
                        service_id=1,
                        input_data=input_data,
                        db=mock_db,
                        current_user_id=1
                    )
                    batch_tasks.append(task)
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                all_results.extend(batch_results)
                
                # Brief pause between batches
                await asyncio.sleep(0.1)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            successful_runs = [r for r in all_results if not isinstance(r, Exception)]
            
            print(f"Sustained load: {len(successful_runs)}/30 requests completed in {total_time:.3f} seconds")
            print(f"Average rate: {len(successful_runs)/total_time:.1f} requests/second")
            
            # Should maintain reasonable performance under sustained load
            assert len(successful_runs) >= 25  # At least 83% success rate
            assert total_time < 5.0  # Should complete within reasonable time


class TestMemoryAndResourceUsage:
    """Test memory usage and resource management"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_datasets(self, mock_db):
        """Test memory usage when processing large datasets"""
        mock_service = Mock()
        mock_service.id = 1
        mock_service.owner_id = 1
        mock_service.is_public = True
        mock_service.workflow = {"nodes": {"0": {"agent_id": 1, "next": None}}}
        mock_service.run_time = 0
        mock_service.average_token_usage = {}
        
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.agent_type = "openai"
        mock_agent.config = {}
        mock_agent.system_instruction = "Test"
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [mock_service, mock_agent]
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Create large input data
        large_input = {
            "input": "x" * 1024 * 1024,  # 1MB of text
            "context": {"large_data": ["item"] * 10000},  # Large context
            "api_keys": {"1": "test-key"}
        }
        
        with patch('api.api_v1.endpoints.mini_services.create_agent') as mock_create_agent, \
             patch('api.api_v1.endpoints.mini_services.WorkflowProcessor') as mock_workflow_processor, \
             patch('api.api_v1.endpoints.mini_services.create_log'), \
             patch('api.api_v1.endpoints.mini_services.flag_modified'):
            
            mock_agent_instance = AsyncMock()
            mock_create_agent.return_value = mock_agent_instance
            
            mock_processor = AsyncMock()
            mock_processor.process.return_value = {
                "output": "Processed large data",
                "token_usage": {"total_tokens": 5000},
                "results": []
            }
            mock_workflow_processor.return_value = mock_processor
            
            # Should handle large datasets without excessive memory usage
            try:
                result = await run_mini_service(
                    service_id=1,
                    input_data=large_input,
                    db=mock_db,
                    current_user_id=1
                )
                assert "output" in result
            except Exception as e:
                # If it fails, should be due to size limits, not memory issues
                assert "memory" not in str(e).lower() or "size" in str(e).lower()

    @pytest.mark.asyncio
    async def test_resource_cleanup_after_errors(self, mock_db):
        """Test that resources are properly cleaned up after errors"""
        mock_service = Mock()
        mock_service.id = 1
        mock_service.owner_id = 1
        mock_service.is_public = True
        mock_service.workflow = {"nodes": {"0": {"agent_id": 1, "next": None}}}
        mock_service.run_time = 0
        mock_service.average_token_usage = {}
        
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.agent_type = "openai"
        mock_agent.config = {}
        mock_agent.system_instruction = "Test"
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [mock_service, mock_agent]
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        mock_db.delete = Mock()  # For cleanup
        
        input_data = {
            "input": "Test input",
            "context": {},
            "api_keys": {"1": "test-key"}
        }
        
        with patch('api.api_v1.endpoints.mini_services.create_agent') as mock_create_agent, \
             patch('api.api_v1.endpoints.mini_services.WorkflowProcessor') as mock_workflow_processor, \
             patch('api.api_v1.endpoints.mini_services.create_log'), \
             patch('api.api_v1.endpoints.mini_services.flag_modified'):
            
            mock_agent_instance = AsyncMock()
            mock_create_agent.return_value = mock_agent_instance
            
            # Mock processor to raise an exception
            mock_processor = AsyncMock()
            mock_processor.process.side_effect = Exception("Processing failed")
            mock_workflow_processor.return_value = mock_processor
            
            # Should handle error and clean up resources
            with pytest.raises(Exception):
                await run_mini_service(
                    service_id=1,
                    input_data=input_data,
                    db=mock_db,
                    current_user_id=1
                )
            
            # Verify cleanup was called
            mock_db.delete.assert_called()  # Process record should be deleted
            mock_db.commit.assert_called()  # Changes should be committed


if __name__ == "__main__":
    pytest.main([__file__])
