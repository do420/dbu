"""
Test configuration and fixtures for the test suite.
Provides common setup for database testing, mocking, and other shared utilities.
"""
import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from typing import Generator

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert project root at the beginning of sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debug: Print path information
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

# Now import your local modules with error handling
try:
    from db.base_class import Base
    from db.session import get_db
    from models.user import User
    from models.agent import Agent
    from models.mini_service import MiniService
    from models.process import Process
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"Contents of project root: {os.listdir(project_root)}")
    print(f"Contents of db: {os.listdir(os.path.join(project_root, 'db'))}")
    print(f"Contents of models: {os.listdir(os.path.join(project_root, 'models'))}")
    raise

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def db_session():
    """Create a database session for testing."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return Mock()

@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    user = User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password"
    )
    return user

@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    agent = Agent(
        id=1,
        name="Test Agent",
        agent_type="openai",
        config={"model": "gpt-3.5-turbo"},
        system_instruction="You are a helpful assistant",
        owner_id=1,
        is_enhanced=False
    )
    return agent

@pytest.fixture
def sample_mini_service():
    """Create a sample mini service for testing."""
    mini_service = MiniService(
        id=1,
        name="Test Service",
        description="A test service",
        workflow={
            "nodes": {
                "0": {"agent_id": 1, "next": None}
            }
        },
        input_type="text",
        output_type="text",
        owner_id=1,
        average_token_usage={},
        run_time=0,
        is_enhanced=False,
        is_public=False
    )
    return mini_service

@pytest.fixture
def sample_process():
    """Create a sample process for testing."""
    process = Process(
        id=1,
        mini_service_id=1,
        user_id=1,
        total_tokens={"total_tokens": 100}
    )
    return process

@pytest.fixture
def client():
    """Create a test client."""
    from main import app
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

@pytest.fixture
def mock_workflow_processor():
    """Create a mock workflow processor."""
    processor = AsyncMock()
    processor.process.return_value = {
        "output": "Test output",
        "token_usage": {"total_tokens": 50},
        "results": []
    }
    return processor

@pytest.fixture
def mock_agent_instance():
    """Create a mock agent instance."""
    agent = AsyncMock()
    agent.process.return_value = {
        "output": "Agent output",
        "token_usage": {"total_tokens": 25}
    }
    return agent

@pytest.fixture
def valid_api_keys():
    """Valid API keys for testing."""
    return {
        "1": "sk-test-openai-key",
        "2": "test-gemini-key"
    }

@pytest.fixture
def sample_workflow():
    """Sample workflow configuration."""
    return {
        "nodes": {
            "0": {"agent_id": 1, "next": 1},
            "1": {"agent_id": 2, "next": None}
        }
    }

@pytest.fixture
def sample_input_data():
    """Sample input data for mini service execution."""
    return {
        "input": "Hello, world!",
        "context": {"test": True},
        "api_keys": {"1": "sk-test-key"}
    }

# Utility functions for testing
def create_mock_agent(agent_id: int = 1, agent_type: str = "openai", is_enhanced: bool = False):
    """Create a mock agent with specified properties."""
    agent = Mock(spec=Agent)
    agent.id = agent_id
    agent.agent_type = agent_type
    agent.is_enhanced = is_enhanced
    agent.config = {"model": "gpt-3.5-turbo"}
    agent.system_instruction = "You are a helpful assistant"
    return agent

def create_mock_mini_service(
    service_id: int = 1, 
    owner_id: int = 1, 
    is_public: bool = False, 
    workflow: dict = None
):
    """Create a mock mini service with specified properties."""
    if workflow is None:
        workflow = {"nodes": {"0": {"agent_id": 1, "next": None}}}
    
    service = Mock(spec=MiniService)
    service.id = service_id
    service.name = "Test Service"
    service.description = "A test service"
    service.owner_id = owner_id
    service.is_public = is_public
    service.workflow = workflow
    service.input_type = "text"
    service.output_type = "text"
    service.run_time = 0
    service.average_token_usage = {}
    service.is_enhanced = False
    return service

# Custom assertions for testing
def assert_http_exception(exception, expected_status_code: int, expected_message: str = None):
    """Assert that an HTTPException has the expected status code and message."""
    assert exception.status_code == expected_status_code
    if expected_message:
        assert expected_message in str(exception.detail)

def assert_database_calls(mock_db, expected_add_calls: int = None, expected_commit_calls: int = None):
    """Assert that the expected database calls were made."""
    if expected_add_calls is not None:
        assert mock_db.add.call_count == expected_add_calls
    if expected_commit_calls is not None:
        assert mock_db.commit.call_count == expected_commit_calls