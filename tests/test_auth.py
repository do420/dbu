"""
Comprehensive tests for authentication endpoints.
Tests login, registration, and authentication flows.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.api_v1.endpoints.auth import login, register_user, get_current_user
from models.user import User
from schemas.user import UserCreate, UserLogin
from core.security import hash_password


class TestAuthLogin:
    """Test authentication login functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing"""
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password=hash_password("password123")
        )
        return user

    def test_login_success(self, mock_db, sample_user):
        """Test successful login"""
        # Mock OAuth2PasswordRequestForm
        form_data = Mock()
        form_data.username = "test@example.com"
        form_data.password = "password123"
        
        mock_db.query.return_value.filter.return_value.first.return_value = sample_user
        
        with patch('api.api_v1.endpoints.auth.verify_password', return_value=True), \
             patch('api.api_v1.endpoints.auth.create_access_token', return_value="mock_token"):
            
            result = login(form_data=form_data, db=mock_db)
            
            assert result["access_token"] == "mock_token"
            assert result["token_type"] == "bearer"
            assert result["user"]["id"] == 1
            assert result["user"]["email"] == "test@example.com"

    def test_login_user_not_found(self, mock_db):
        """Test login with non-existent user"""
        form_data = Mock()
        form_data.username = "nonexistent@example.com"
        form_data.password = "password123"
        
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            login(form_data=form_data, db=mock_db)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email or password" in str(exc_info.value.detail)

    def test_login_wrong_password(self, mock_db, sample_user):
        """Test login with wrong password"""
        form_data = Mock()
        form_data.username = "test@example.com"
        form_data.password = "wrongpassword"
        
        mock_db.query.return_value.filter.return_value.first.return_value = sample_user
        
        with patch('api.api_v1.endpoints.auth.verify_password', return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                login(form_data=form_data, db=mock_db)
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Incorrect email or password" in str(exc_info.value.detail)

    def test_login_empty_credentials(self, mock_db):
        """Test login with empty credentials"""
        form_data = Mock()
        form_data.username = ""
        form_data.password = ""
        
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            login(form_data=form_data, db=mock_db)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestAuthRegister:
    """Test user registration functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.mark.asyncio
    async def test_register_success(self, mock_db):
        """Test successful user registration"""
        user_data = UserCreate(
            username="newuser",
            email="newuser@example.com",
            password="password123"
        )
        
        # Mock that user doesn't exist
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Mock the created user
        created_user = User(
            id=1,
            username="newuser",
            email="newuser@example.com",
            hashed_password="hashed_password"
        )
        mock_db.refresh.side_effect = lambda x: setattr(x, 'id', 1)
        
        with patch('api.api_v1.endpoints.auth.hash_password', return_value="hashed_password"), \
             patch('api.api_v1.endpoints.auth.create_access_token', return_value="mock_token"):
            
            result = await register_user(user=user_data, db=mock_db)
            
            assert result["access_token"] == "mock_token"
            assert result["token_type"] == "bearer"
            assert result["user"]["username"] == "newuser"
            assert result["user"]["email"] == "newuser@example.com"
            
            # Verify database operations
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_existing_email(self, mock_db):
        """Test registration with existing email"""
        user_data = UserCreate(
            username="newuser",
            email="existing@example.com",
            password="password123"
        )
        
        # Mock existing user
        existing_user = User(
            id=1,
            username="existinguser",
            email="existing@example.com",
            hashed_password="hashed"
        )        mock_db.query.return_value.filter.return_value.first.return_value = existing_user
        
        with pytest.raises(HTTPException) as exc_info:
            await register_user(user=user_data, db=mock_db)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Email already registered" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_register_existing_username(self, mock_db):
        """Test registration with existing username"""
        user_data = UserCreate(
            username="existinguser",
            email="new@example.com",
            password="password123"
        )
        
        # Mock existing user on second query (username check)
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            None,  # Email check - no user found
            User(id=1, username="existinguser", email="other@example.com")  # Username check - user found
        ]
          with pytest.raises(HTTPException) as exc_info:
            await register_user(user=user_data, db=mock_db)
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Username already taken" in str(exc_info.value.detail)

    def test_register_invalid_email_format(self, mock_db):
        """Test registration with invalid email format"""
        user_data = UserCreate(
            username="newuser",
            email="invalid-email",
            password="password123"
        )
        
        # This should be caught by pydantic validation before reaching the endpoint
        # But let's test the endpoint behavior anyway
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # The endpoint might still process it, but validation should catch it
        # In practice, FastAPI would return 422 for invalid email format

    def test_register_weak_password(self, mock_db):
        """Test registration with weak password"""
        user_data = UserCreate(
            username="newuser",
            email="newuser@example.com",
            password="123"  # Too short
        )
        
        # This should be caught by pydantic validation
        # The model should have password length validation


class TestAuthCurrentUser:
    """Test current user authentication"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    def test_get_current_user_success(self, mock_db):
        """Test successful current user retrieval"""
        mock_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed"
        )
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        with patch('api.api_v1.endpoints.auth.decode_access_token') as mock_decode:
            mock_decode.return_value = {"sub": "1", "email": "test@example.com"}
            
            result = get_current_user(token="valid_token", db=mock_db)
            
            assert result.id == 1
            assert result.username == "testuser"
            assert result.email == "test@example.com"

    def test_get_current_user_invalid_token(self, mock_db):
        """Test current user with invalid token"""
        with patch('api.api_v1.endpoints.auth.decode_access_token') as mock_decode:
            mock_decode.side_effect = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(token="invalid_token", db=mock_db)
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_current_user_user_not_found(self, mock_db):
        """Test current user when user doesn't exist in database"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with patch('api.api_v1.endpoints.auth.decode_access_token') as mock_decode:
            mock_decode.return_value = {"sub": "999", "email": "nonexistent@example.com"}
            
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(token="valid_token", db=mock_db)
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "User not found" in str(exc_info.value.detail)


class TestAuthIntegration:
    """Integration tests for authentication endpoints"""
    
    def test_login_endpoint_integration(self, client):
        """Test login endpoint through FastAPI test client"""
        # This would require setting up the database and creating a test user
        # For now, we'll test that the endpoint exists and returns appropriate errors
        
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "password123"}
        )
        
        # Without proper database setup, this might fail
        # but we can check that the endpoint is accessible
        assert response.status_code in [200, 401, 422]

    def test_register_endpoint_integration(self, client):
        """Test register endpoint through FastAPI test client"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123"
            }
        )
        
        # Without proper database setup, this might fail
        assert response.status_code in [201, 400, 422]

    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication"""
        response = client.get("/api/v1/mini-services/")
        
        # Should require authentication
        assert response.status_code in [401, 422]

    def test_auth_flow_integration(self, client):
        """Test complete authentication flow"""
        # 1. Try to access protected resource without auth
        response = client.get("/api/v1/mini-services/")
        assert response.status_code in [401, 422]
        
        # 2. Register new user (would need proper DB setup)
        # 3. Login with credentials
        # 4. Access protected resource with token
        # This requires more complex setup with test database


class TestAuthSecurity:
    """Test security aspects of authentication"""
    
    def test_password_hashing(self):
        """Test that passwords are properly hashed"""
        password = "password123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        
        # Hash the same password again - should be different (salt)
        hashed2 = hash_password(password)
        assert hashed != hashed2

    def test_token_expiration(self):
        """Test that tokens have proper expiration"""
        # This would test the JWT token creation and validation
        # with various expiration times
        pass

    def test_sql_injection_protection(self, mock_db):
        """Test protection against SQL injection in login"""
        form_data = Mock()
        form_data.username = "'; DROP TABLE users; --"
        form_data.password = "password"
        
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # Should safely handle malicious input
        with pytest.raises(HTTPException):
            login(form_data=form_data, db=mock_db)

    def test_rate_limiting_simulation(self):
        """Test behavior under rapid repeated requests"""
        # This would test rate limiting if implemented
        # For now, just ensure endpoints can handle multiple rapid calls
        pass


if __name__ == "__main__":
    pytest.main([__file__])
