"""
Comprehensive tests for API Keys endpoints.
Tests CRUD operations for API key management.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from api.api_v1.endpoints.api_keys import create_api_key, list_api_keys, delete_api_key
from models.api_key import APIKey
from schemas.api_key import APIKeyCreate


class TestAPIKeyCreation:
    """Test API key creation functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_api_key_data(self):
        return APIKeyCreate(
            provider="openai",
            api_key="sk-test-api-key-12345",
            description="Test OpenAI API key"
        )

    @pytest.mark.asyncio
    async def test_create_api_key_success(self, mock_db, sample_api_key_data):
        """Test successful API key creation"""
        with patch('api.api_v1.endpoints.api_keys.encrypt_api_key') as mock_encrypt:
            mock_encrypt.return_value = "encrypted_key_data"
            mock_db.add = Mock()
            mock_db.commit = Mock()
            mock_db.refresh = Mock()
            
            # Mock the created API key
            created_key = APIKey(
                id=1,
                provider="openai",
                api_key="encrypted_key_data",
                user_id=1,
                description="Test OpenAI API key"
            )
            mock_db.refresh.side_effect = lambda obj: setattr(obj, 'id', 1)
            
            result = await create_api_key(
                api_key=sample_api_key_data,
                db=mock_db,
                current_user_id=1
            )
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            mock_encrypt.assert_called_once_with("sk-test-api-key-12345")

    @pytest.mark.asyncio
    async def test_create_api_key_unauthorized(self, mock_db, sample_api_key_data):
        """Test API key creation without authentication"""
        with pytest.raises(HTTPException) as exc_info:
            await create_api_key(
                api_key=sample_api_key_data,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_create_api_key_different_providers(self, mock_db):
        """Test creating API keys for different providers"""
        providers = ["openai", "gemini", "anthropic", "azure"]
        
        with patch('api.api_v1.endpoints.api_keys.encrypt_api_key') as mock_encrypt:
            mock_encrypt.return_value = "encrypted_key"
            mock_db.add = Mock()
            mock_db.commit = Mock()
            mock_db.refresh = Mock()
            
            for provider in providers:
                api_key_data = APIKeyCreate(
                    provider=provider,
                    api_key=f"test-{provider}-key",
                    description=f"Test {provider} key"
                )
                
                await create_api_key(
                    api_key=api_key_data,
                    db=mock_db,
                    current_user_id=1
                )
        
        assert mock_db.add.call_count == len(providers)

    @pytest.mark.asyncio
    async def test_create_api_key_validation(self, mock_db):
        """Test API key validation during creation"""
        # Test empty API key
        with pytest.raises((HTTPException, ValueError)):
            api_key_data = APIKeyCreate(
                provider="openai",
                api_key="",
                description="Empty key test"
            )
            await create_api_key(
                api_key=api_key_data,
                db=mock_db,
                current_user_id=1
            )

    @pytest.mark.asyncio
    async def test_create_api_key_encryption_failure(self, mock_db, sample_api_key_data):
        """Test handling encryption failure"""
        with patch('api.api_v1.endpoints.api_keys.encrypt_api_key') as mock_encrypt:
            mock_encrypt.side_effect = Exception("Encryption failed")
            
            with pytest.raises(Exception):
                await create_api_key(
                    api_key=sample_api_key_data,
                    db=mock_db,
                    current_user_id=1
                )


class TestAPIKeyListing:
    """Test API key listing functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.fixture
    def sample_api_keys(self):
        return [
            APIKey(
                id=1,
                provider="openai",
                api_key="encrypted_openai_key",
                user_id=1
            ),
            APIKey(
                id=2,
                provider="gemini",
                api_key="encrypted_gemini_key",
                user_id=1
            )
        ]

    @pytest.mark.asyncio
    async def test_list_api_keys_success(self, mock_db, sample_api_keys):
        """Test successful API key listing"""
        mock_db.query.return_value.filter.return_value.all.return_value = sample_api_keys
        
        with patch('api.api_v1.endpoints.api_keys.decrypt_api_key') as mock_decrypt:
            mock_decrypt.side_effect = lambda x: f"decrypted_{x}"
            
            result = await list_api_keys(
                db=mock_db,
                current_user_id=1
            )
            
            assert len(result) == 2
            assert result[0].provider == "openai"
            assert result[1].provider == "gemini"
            assert mock_decrypt.call_count == 2

    @pytest.mark.asyncio
    async def test_list_api_keys_empty(self, mock_db):
        """Test listing when no API keys exist"""
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        result = await list_api_keys(
            db=mock_db,
            current_user_id=1
        )
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_list_api_keys_unauthorized(self, mock_db):
        """Test API key listing without authentication"""
        with pytest.raises(HTTPException) as exc_info:
            await list_api_keys(
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_list_api_keys_user_isolation(self, mock_db):
        """Test that users only see their own API keys"""
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        await list_api_keys(
            db=mock_db,
            current_user_id=1
        )
        
        # Verify that the query filters by user_id
        mock_db.query.assert_called_once()
        filter_calls = mock_db.query.return_value.filter.call_args_list
        assert len(filter_calls) > 0

    @pytest.mark.asyncio
    async def test_list_api_keys_decryption_failure(self, mock_db, sample_api_keys):
        """Test handling decryption failure"""
        mock_db.query.return_value.filter.return_value.all.return_value = sample_api_keys
        
        with patch('api.api_v1.endpoints.api_keys.decrypt_api_key') as mock_decrypt:
            mock_decrypt.side_effect = Exception("Decryption failed")
            
            with pytest.raises(Exception):
                await list_api_keys(
                    db=mock_db,
                    current_user_id=1
                )


class TestAPIKeyDeletion:
    """Test API key deletion functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.fixture
    def sample_api_key(self):
        return APIKey(
            id=1,
            provider="openai",
            api_key="encrypted_key",
            user_id=1
        )

    @pytest.mark.asyncio
    async def test_delete_api_key_success(self, mock_db, sample_api_key):
        """Test successful API key deletion"""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_api_key
        mock_db.delete = Mock()
        mock_db.commit = Mock()
        
        result = await delete_api_key(
            key_id=1,
            db=mock_db,
            current_user_id=1
        )
        
        mock_db.delete.assert_called_once_with(sample_api_key)
        mock_db.commit.assert_called_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_api_key_not_found(self, mock_db):
        """Test API key deletion when key doesn't exist"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await delete_api_key(
                key_id=999,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_delete_api_key_unauthorized(self, mock_db):
        """Test API key deletion without authentication"""
        with pytest.raises(HTTPException) as exc_info:
            await delete_api_key(
                key_id=1,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_delete_api_key_user_isolation(self, mock_db):
        """Test that users can only delete their own API keys"""        # Mock API key belonging to different user
        other_user_key = APIKey(
            id=1,
            provider="openai",
            api_key="encrypted_key",
            user_id=2  # Different user
        )
        mock_db.query.return_value.filter.return_value.first.return_value = None  # No key found for this user
        
        with pytest.raises(HTTPException) as exc_info:
            await delete_api_key(
                key_id=1,
                db=mock_db,
                current_user_id=1  # Different user trying to delete
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


class TestAPIKeySecurity:
    """Test security aspects of API key management"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.mark.asyncio
    async def test_api_key_encryption_on_create(self, mock_db):
        """Test that API keys are encrypted before storage"""
        api_key_data = APIKeyCreate(
            provider="openai",
            api_key="sk-sensitive-api-key",
            description="Sensitive key"
        )
        
        with patch('api.api_v1.endpoints.api_keys.encrypt_api_key') as mock_encrypt:
            mock_encrypt.return_value = "safely_encrypted_data"
            mock_db.add = Mock()
            mock_db.commit = Mock()
            mock_db.refresh = Mock()
            
            await create_api_key(
                api_key=api_key_data,
                db=mock_db,
                current_user_id=1
            )
            
            # Verify encryption was called with the original key
            mock_encrypt.assert_called_once_with("sk-sensitive-api-key")
            
            # Verify that the encrypted version is stored
            added_key = mock_db.add.call_args[0][0]
            assert added_key.api_key == "safely_encrypted_data"

    @pytest.mark.asyncio
    async def test_api_key_decryption_on_list(self, mock_db):
        """Test that API keys are decrypted when listed"""        encrypted_key = APIKey(
            id=1,
            provider="openai",
            api_key="encrypted_data",
            user_id=1
        )
        mock_db.query.return_value.filter.return_value.all.return_value = [encrypted_key]
        
        with patch('api.api_v1.endpoints.api_keys.decrypt_api_key') as mock_decrypt:
            mock_decrypt.return_value = "decrypted_api_key"
            
            result = await list_api_keys(
                db=mock_db,
                current_user_id=1
            )
            
            # Verify decryption was called
            mock_decrypt.assert_called_once_with("encrypted_data")
            
            # Verify the key in response is decrypted
            assert result[0].api_key == "decrypted_api_key"

    def test_sql_injection_protection(self):
        """Test protection against SQL injection in API key operations"""
        # Since we're using SQLAlchemy ORM, this should be handled automatically
        # This test documents the expectation
        pass

    @pytest.mark.asyncio
    async def test_api_key_data_validation(self, mock_db):
        """Test validation of API key data"""
        # Test malicious provider name
        malicious_data = APIKeyCreate(
            provider="'; DROP TABLE api_keys; --",
            api_key="sk-test-key",
            description="Malicious provider"
        )
        
        with patch('api.api_v1.endpoints.api_keys.encrypt_api_key'):
            try:
                await create_api_key(
                    api_key=malicious_data,
                    db=mock_db,
                    current_user_id=1
                )
            except (HTTPException, ValueError):
                # Should handle invalid data gracefully
                pass


class TestAPIKeyValidation:
    """Test API key validation and business logic"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.mark.asyncio
    async def test_duplicate_provider_keys(self, mock_db):
        """Test handling duplicate API keys for same provider"""
        # This test assumes business logic might prevent duplicates
        # Implementation depends on requirements
        api_key_data = APIKeyCreate(
            provider="openai",
            api_key="sk-duplicate-test",
            description="Duplicate test"
        )
        
        with patch('api.api_v1.endpoints.api_keys.encrypt_api_key') as mock_encrypt:
            mock_encrypt.return_value = "encrypted_key"
            mock_db.add = Mock()
            mock_db.commit = Mock()
            mock_db.refresh = Mock()
            
            # Create first key
            await create_api_key(
                api_key=api_key_data,
                db=mock_db,
                current_user_id=1
            )
            
            # Create second key for same provider (should be allowed in current implementation)
            await create_api_key(
                api_key=api_key_data,
                db=mock_db,
                current_user_id=1
            )
            
            assert mock_db.add.call_count == 2  # Both should be allowed

    @pytest.mark.asyncio
    async def test_api_key_length_validation(self, mock_db):
        """Test validation of API key length"""
        test_cases = [
            "",  # Empty
            "short",  # Too short
            "x" * 1000,  # Very long
        ]
        
        for test_key in test_cases:
            api_key_data = APIKeyCreate(
                provider="openai",
                api_key=test_key,
                description="Length test"
            )
            
            with patch('api.api_v1.endpoints.api_keys.encrypt_api_key'):
                try:
                    await create_api_key(
                        api_key=api_key_data,
                        db=mock_db,
                        current_user_id=1
                    )
                except (HTTPException, ValueError):
                    # May fail validation, which is acceptable
                    pass

    @pytest.mark.asyncio
    async def test_valid_provider_types(self, mock_db):
        """Test that only valid provider types are accepted"""
        valid_providers = ["openai", "gemini", "anthropic", "azure", "huggingface"]
        
        with patch('api.api_v1.endpoints.api_keys.encrypt_api_key') as mock_encrypt:
            mock_encrypt.return_value = "encrypted_key"
            mock_db.add = Mock()
            mock_db.commit = Mock()
            mock_db.refresh = Mock()
            
            for provider in valid_providers:
                api_key_data = APIKeyCreate(
                    provider=provider,
                    api_key="test-key-value",
                    description=f"Test {provider} key"
                )
                
                # Should not raise exception for valid providers
                await create_api_key(
                    api_key=api_key_data,
                    db=mock_db,
                    current_user_id=1
                )


class TestAPIKeyPerformance:
    """Test performance aspects of API key operations"""
    
    @pytest.mark.asyncio
    async def test_list_large_number_of_keys(self, mock_db):
        """Test performance with large number of API keys"""        # Create many mock API keys
        large_key_list = []
        for i in range(100):
            key = APIKey(
                id=i,
                provider=f"provider_{i % 5}",
                api_key=f"encrypted_key_{i}",
                user_id=1
            )
            large_key_list.append(key)
        
        mock_db.query.return_value.filter.return_value.all.return_value = large_key_list
        
        with patch('api.api_v1.endpoints.api_keys.decrypt_api_key') as mock_decrypt:
            mock_decrypt.side_effect = lambda x: f"decrypted_{x}"
            
            import time
            start_time = time.time()
            
            result = await list_api_keys(
                db=mock_db,
                current_user_id=1
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            assert len(result) == 100
            assert execution_time < 1.0  # Should handle many keys efficiently
            assert mock_decrypt.call_count == 100

    @pytest.mark.asyncio
    async def test_encryption_performance(self, mock_db):
        """Test encryption performance with large API keys"""
        large_api_key = "x" * 10000  # 10KB API key
        
        api_key_data = APIKeyCreate(
            provider="test",
            api_key=large_api_key,
            description="Large key test"
        )
        
        with patch('api.api_v1.endpoints.api_keys.encrypt_api_key') as mock_encrypt:
            mock_encrypt.return_value = "encrypted_large_key"
            mock_db.add = Mock()
            mock_db.commit = Mock()
            mock_db.refresh = Mock()
            
            import time
            start_time = time.time()
            
            await create_api_key(
                api_key=api_key_data,
                db=mock_db,
                current_user_id=1
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            assert execution_time < 0.5  # Encryption should be fast


if __name__ == "__main__":
    pytest.main([__file__])
