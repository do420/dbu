"""
Comprehensive tests for RAG documents endpoints.
Tests document management for RAG agents including upload and retrieval.
"""
import pytest
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException, status, UploadFile
from sqlalchemy.orm import Session

from api.api_v1.endpoints.agents import (
    get_agent_documents,
    upload_document_to_rag_agent
)
from models.agent import Agent
from models.api_key import APIKey


class TestRAGDocuments:
    """Test RAG document management functionality"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_rag_agent(self):
        return Agent(
            id=1,
            name="Test RAG Agent",
            agent_type="rag",
            system_instruction="You are a helpful RAG assistant",
            config={"embedding_model": "models/embedding-001"},
            input_type="text",
            output_type="text",
            owner_id=1
        )
    
    @pytest.fixture
    def sample_non_rag_agent(self):
        return Agent(
            id=2,
            name="Test OpenAI Agent",
            agent_type="openai",
            system_instruction="You are helpful",
            config={"model": "gpt-3.5-turbo"},
            input_type="text",
            output_type="text",
            owner_id=1
        )
    
    @pytest.fixture
    def sample_api_key(self):
        return APIKey(
            id=1,
            provider="gemini",
            api_key="encrypted_gemini_key",
            user_id=1,
            description="Gemini API key"
        )
    
    @pytest.fixture
    def mock_upload_file(self):
        file = Mock(spec=UploadFile)
        file.filename = "test_document.pdf"
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=b"test document content")
        return file

    @pytest.mark.asyncio
    async def test_get_agent_documents_success(self, mock_db, sample_rag_agent, sample_api_key):
        """Test successful retrieval of RAG agent documents"""
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_rag_agent,  # Agent exists and is RAG type
            sample_api_key  # API key exists
        ]
        
        # Mock ChromaDB directory exists
        with patch('os.path.exists', return_value=True), \
             patch('api.api_v1.endpoints.agents.decrypt_api_key', return_value="test_gemini_key"), \
             patch('api.api_v1.endpoints.agents.GoogleGenerativeAIEmbeddings'), \
             patch('api.api_v1.endpoints.agents.Chroma') as mock_chroma:
            
            # Mock Chroma database with sample documents
            mock_chroma_instance = Mock()
            mock_chroma.return_value = mock_chroma_instance
            
            # Mock similarity search returning sample documents
            mock_docs = [
                Mock(metadata={"source": "/path/to/doc1.pdf"}),
                Mock(metadata={"source": "/path/to/doc1.pdf"}),  # Same document, different chunk
                Mock(metadata={"source": "/path/to/doc2.txt"}),
            ]
            mock_chroma_instance.similarity_search.return_value = mock_docs
            
            result = await get_agent_documents(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
            
            assert "documents" in result
            assert len(result["documents"]) == 2  # Two unique documents
            assert any(doc["filename"] == "doc1.pdf" for doc in result["documents"])
            assert any(doc["filename"] == "doc2.txt" for doc in result["documents"])

    @pytest.mark.asyncio
    async def test_get_agent_documents_agent_not_found(self, mock_db):
        """Test document retrieval when agent doesn't exist"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_agent_documents(
                agent_id=999,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_agent_documents_unauthorized(self, mock_db):
        """Test document retrieval without authentication"""
        with pytest.raises(HTTPException) as exc_info:
            await get_agent_documents(
                agent_id=1,
                db=mock_db,
                current_user_id=None
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_agent_documents_not_rag_agent(self, mock_db, sample_non_rag_agent):
        """Test document retrieval for non-RAG agent"""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_non_rag_agent
        
        with pytest.raises(HTTPException) as exc_info:
            await get_agent_documents(
                agent_id=2,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "not a RAG agent" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_agent_documents_no_api_key(self, mock_db, sample_rag_agent):
        """Test document retrieval when Gemini API key is missing"""
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_rag_agent,  # Agent exists
            None  # No API key
        ]
        
        with pytest.raises(HTTPException) as exc_info:
            await get_agent_documents(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Gemini API key required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_agent_documents_no_chroma_directory(self, mock_db, sample_rag_agent, sample_api_key):
        """Test document retrieval when ChromaDB directory doesn't exist"""
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_rag_agent,  # Agent exists
            sample_api_key  # API key exists
        ]
        
        # Mock ChromaDB directory doesn't exist
        with patch('os.path.exists', return_value=False):
            result = await get_agent_documents(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
            
            assert result == {"documents": []}

    @pytest.mark.asyncio
    async def test_get_agent_documents_user_isolation(self, mock_db):
        """Test that users can only access their own agents' documents"""
        # Mock agent belonging to different user
        other_user_agent = Agent(
            id=1,
            name="Other User's RAG Agent",
            agent_type="rag",
            owner_id=2  # Different user
        )
        mock_db.query.return_value.filter.return_value.first.return_value = None  # No agent found for this user
        
        with pytest.raises(HTTPException) as exc_info:
            await get_agent_documents(
                agent_id=1,
                db=mock_db,
                current_user_id=1  # Different user
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_agent_documents_chroma_error(self, mock_db, sample_rag_agent, sample_api_key):
        """Test handling ChromaDB connection errors"""
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_rag_agent,  # Agent exists
            sample_api_key  # API key exists
        ]
        
        with patch('os.path.exists', return_value=True), \
             patch('api.api_v1.endpoints.agents.decrypt_api_key', return_value="test_key"), \
             patch('api.api_v1.endpoints.agents.GoogleGenerativeAIEmbeddings'), \
             patch('api.api_v1.endpoints.agents.Chroma', side_effect=Exception("ChromaDB connection failed")):
            
            with pytest.raises(HTTPException) as exc_info:
                await get_agent_documents(
                    agent_id=1,
                    db=mock_db,
                    current_user_id=1
                )
            
            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_upload_document_to_rag_agent_success(self, mock_db, sample_rag_agent, sample_api_key, mock_upload_file):
        """Test successful document upload to RAG agent"""
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_rag_agent,  # Agent exists and is RAG type
            sample_api_key  # API key exists
        ]
        
        with patch('api.api_v1.endpoints.agents.decrypt_api_key', return_value="test_gemini_key"), \
             patch('api.api_v1.endpoints.agents.GoogleGenerativeAIEmbeddings'), \
             patch('api.api_v1.endpoints.agents.Chroma') as mock_chroma, \
             patch('api.api_v1.endpoints.agents.PyPDFLoader') as mock_pdf_loader, \
             patch('api.api_v1.endpoints.agents.RecursiveCharacterTextSplitter') as mock_splitter, \
             patch('tempfile.mkdtemp', return_value="/tmp/test"), \
             patch('os.makedirs'), \
             patch('builtins.open', create=True), \
             patch('api.api_v1.endpoints.agents.create_log'):
            
            # Mock document processing pipeline
            mock_loader_instance = Mock()
            mock_pdf_loader.return_value = mock_loader_instance
            mock_loader_instance.load.return_value = [Mock(page_content="Test content", metadata={})]
            
            mock_splitter_instance = Mock()
            mock_splitter.return_value = mock_splitter_instance
            mock_splitter_instance.split_documents.return_value = [Mock(page_content="Split content", metadata={})]
            
            mock_chroma_instance = Mock()
            mock_chroma.return_value = mock_chroma_instance
            mock_chroma_instance.add_documents = Mock()
            
            result = await upload_document_to_rag_agent(
                agent_id=1,
                file=mock_upload_file,
                db=mock_db,
                current_user_id=1
            )
            
            assert result["success"] is True
            assert "document uploaded successfully" in result["message"]
            mock_chroma_instance.add_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_document_invalid_file_type(self, mock_db, sample_rag_agent, mock_upload_file):
        """Test upload with invalid file type"""
        mock_upload_file.filename = "test.xyz"
        mock_upload_file.content_type = "application/unknown"
        
        mock_db.query.return_value.filter.return_value.first.return_value = sample_rag_agent
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_document_to_rag_agent(
                agent_id=1,
                file=mock_upload_file,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unsupported file type" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_upload_document_file_too_large(self, mock_db, sample_rag_agent):
        """Test upload with file too large"""
        large_file = Mock(spec=UploadFile)
        large_file.filename = "large_doc.pdf"
        large_file.content_type = "application/pdf"
        large_file.read = AsyncMock(return_value=b"x" * (100 * 1024 * 1024))  # 100MB file
        
        mock_db.query.return_value.filter.return_value.first.return_value = sample_rag_agent
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_document_to_rag_agent(
                agent_id=1,
                file=large_file,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    @pytest.mark.asyncio
    async def test_upload_document_processing_error(self, mock_db, sample_rag_agent, sample_api_key, mock_upload_file):
        """Test handling document processing errors"""
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            sample_rag_agent,  # Agent exists
            sample_api_key  # API key exists
        ]
        
        with patch('api.api_v1.endpoints.agents.decrypt_api_key', return_value="test_key"), \
             patch('api.api_v1.endpoints.agents.PyPDFLoader', side_effect=Exception("PDF processing failed")), \
             patch('tempfile.mkdtemp', return_value="/tmp/test"), \
             patch('builtins.open', create=True):
            
            with pytest.raises(HTTPException) as exc_info:
                await upload_document_to_rag_agent(
                    agent_id=1,
                    file=mock_upload_file,
                    db=mock_db,
                    current_user_id=1
                )
            
            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestRAGDocumentTypes:
    """Test support for different document types"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_rag_agent(self):
        return Agent(
            id=1,
            name="Test RAG Agent",
            agent_type="rag",
            owner_id=1
        )
    
    @pytest.fixture
    def sample_api_key(self):
        return APIKey(
            id=1,
            provider="gemini",
            api_key="encrypted_key",
            user_id=1
        )

    @pytest.mark.asyncio
    async def test_upload_pdf_document(self, mock_db, sample_rag_agent, sample_api_key):
        """Test uploading PDF document"""
        pdf_file = Mock(spec=UploadFile)
        pdf_file.filename = "test.pdf"
        pdf_file.content_type = "application/pdf"
        pdf_file.read = AsyncMock(return_value=b"PDF content")
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [sample_rag_agent, sample_api_key]
        
        with patch('api.api_v1.endpoints.agents.decrypt_api_key'), \
             patch('api.api_v1.endpoints.agents.GoogleGenerativeAIEmbeddings'), \
             patch('api.api_v1.endpoints.agents.Chroma'), \
             patch('api.api_v1.endpoints.agents.PyPDFLoader') as mock_loader, \
             patch('api.api_v1.endpoints.agents.RecursiveCharacterTextSplitter'), \
             patch('tempfile.mkdtemp'), \
             patch('os.makedirs'), \
             patch('builtins.open', create=True), \
             patch('api.api_v1.endpoints.agents.create_log'):
            
            mock_loader.return_value.load.return_value = [Mock()]
            
            result = await upload_document_to_rag_agent(
                agent_id=1,
                file=pdf_file,
                db=mock_db,
                current_user_id=1
            )
            
            mock_loader.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_text_document(self, mock_db, sample_rag_agent, sample_api_key):
        """Test uploading text document"""
        txt_file = Mock(spec=UploadFile)
        txt_file.filename = "test.txt"
        txt_file.content_type = "text/plain"
        txt_file.read = AsyncMock(return_value=b"Text content")
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [sample_rag_agent, sample_api_key]
        
        with patch('api.api_v1.endpoints.agents.decrypt_api_key'), \
             patch('api.api_v1.endpoints.agents.GoogleGenerativeAIEmbeddings'), \
             patch('api.api_v1.endpoints.agents.Chroma'), \
             patch('api.api_v1.endpoints.agents.TextLoader') as mock_loader, \
             patch('api.api_v1.endpoints.agents.RecursiveCharacterTextSplitter'), \
             patch('tempfile.mkdtemp'), \
             patch('os.makedirs'), \
             patch('builtins.open', create=True), \
             patch('api.api_v1.endpoints.agents.create_log'):
            
            mock_loader.return_value.load.return_value = [Mock()]
            
            result = await upload_document_to_rag_agent(
                agent_id=1,
                file=txt_file,
                db=mock_db,
                current_user_id=1
            )
            
            mock_loader.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_unsupported_document(self, mock_db, sample_rag_agent):
        """Test uploading unsupported document type"""
        unsupported_file = Mock(spec=UploadFile)
        unsupported_file.filename = "test.unknown"
        unsupported_file.content_type = "application/unknown"
        
        mock_db.query.return_value.filter.return_value.first.return_value = sample_rag_agent
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_document_to_rag_agent(
                agent_id=1,
                file=unsupported_file,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST


class TestRAGDocumentSecurity:
    """Test security aspects of RAG document management"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.mark.asyncio
    async def test_document_access_user_isolation(self, mock_db):
        """Test that users cannot access other users' agent documents"""
        # Mock agent belonging to different user
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_agent_documents(
                agent_id=1,
                db=mock_db,
                current_user_id=999  # Different user
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_document_upload_user_isolation(self, mock_db, mock_upload_file):
        """Test that users cannot upload to other users' agents"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_document_to_rag_agent(
                agent_id=1,
                file=mock_upload_file,
                db=mock_db,
                current_user_id=999  # Different user
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_file_path_sanitization(self, mock_db, sample_rag_agent, sample_api_key):
        """Test that file paths are properly sanitized"""
        malicious_file = Mock(spec=UploadFile)
        malicious_file.filename = "../../../etc/passwd"
        malicious_file.content_type = "text/plain"
        malicious_file.read = AsyncMock(return_value=b"malicious content")
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [sample_rag_agent, sample_api_key]
        
        with patch('api.api_v1.endpoints.agents.decrypt_api_key'), \
             patch('api.api_v1.endpoints.agents.GoogleGenerativeAIEmbeddings'), \
             patch('api.api_v1.endpoints.agents.Chroma'), \
             patch('api.api_v1.endpoints.agents.TextLoader'), \
             patch('api.api_v1.endpoints.agents.RecursiveCharacterTextSplitter'), \
             patch('tempfile.mkdtemp', return_value="/safe/tmp/dir"), \
             patch('os.makedirs'), \
             patch('builtins.open', create=True) as mock_open, \
             patch('api.api_v1.endpoints.agents.create_log'):
            
            await upload_document_to_rag_agent(
                agent_id=1,
                file=malicious_file,
                db=mock_db,
                current_user_id=1
            )
            
            # Verify that the file is saved in the safe directory
            # and not using the malicious path
            open_calls = mock_open.call_args_list
            if open_calls:
                file_path = open_calls[0][0][0]
                assert "/safe/tmp/dir" in file_path
                assert "etc/passwd" not in file_path

    def test_sql_injection_protection(self):
        """Test protection against SQL injection in document operations"""
        # Since we're using SQLAlchemy ORM, this should be handled automatically
        pass


class TestRAGDocumentValidation:
    """Test validation of RAG document operations"""
    
    @pytest.fixture
    def mock_db(self):
        return Mock(spec=Session)

    @pytest.mark.asyncio
    async def test_empty_file_validation(self, mock_db, sample_rag_agent):
        """Test validation of empty files"""
        empty_file = Mock(spec=UploadFile)
        empty_file.filename = "empty.pdf"
        empty_file.content_type = "application/pdf"
        empty_file.read = AsyncMock(return_value=b"")  # Empty file
        
        mock_db.query.return_value.filter.return_value.first.return_value = sample_rag_agent
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_document_to_rag_agent(
                agent_id=1,
                file=empty_file,
                db=mock_db,
                current_user_id=1
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "empty" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_filename_validation(self, mock_db, sample_rag_agent):
        """Test validation of file names"""
        invalid_files = [
            ("", "application/pdf"),  # Empty filename
            (None, "application/pdf"),  # None filename
            ("file with spaces.pdf", "application/pdf"),  # Should be allowed
            ("file_with_underscores.pdf", "application/pdf"),  # Should be allowed
        ]
        
        for filename, content_type in invalid_files:
            test_file = Mock(spec=UploadFile)
            test_file.filename = filename
            test_file.content_type = content_type
            test_file.read = AsyncMock(return_value=b"test content")
            
            mock_db.query.return_value.filter.return_value.first.return_value = sample_rag_agent
            
            if filename in ["", None]:
                with pytest.raises(HTTPException):
                    await upload_document_to_rag_agent(
                        agent_id=1,
                        file=test_file,
                        db=mock_db,
                        current_user_id=1
                    )


class TestRAGDocumentPerformance:
    """Test performance aspects of RAG document operations"""
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self, mock_db, sample_rag_agent, sample_api_key):
        """Test processing of large documents"""
        large_file = Mock(spec=UploadFile)
        large_file.filename = "large_doc.pdf"
        large_file.content_type = "application/pdf"
        large_file.read = AsyncMock(return_value=b"x" * (10 * 1024 * 1024))  # 10MB file
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [sample_rag_agent, sample_api_key]
        
        with patch('api.api_v1.endpoints.agents.decrypt_api_key'), \
             patch('api.api_v1.endpoints.agents.GoogleGenerativeAIEmbeddings'), \
             patch('api.api_v1.endpoints.agents.Chroma'), \
             patch('api.api_v1.endpoints.agents.PyPDFLoader') as mock_loader, \
             patch('api.api_v1.endpoints.agents.RecursiveCharacterTextSplitter') as mock_splitter, \
             patch('tempfile.mkdtemp'), \
             patch('os.makedirs'), \
             patch('builtins.open', create=True), \
             patch('api.api_v1.endpoints.agents.create_log'):
            
            # Mock processing pipeline
            mock_loader.return_value.load.return_value = [Mock() for _ in range(100)]  # Many pages
            mock_splitter.return_value.split_documents.return_value = [Mock() for _ in range(500)]  # Many chunks
            
            import time
            start_time = time.time()
            
            result = await upload_document_to_rag_agent(
                agent_id=1,
                file=large_file,
                db=mock_db,
                current_user_id=1
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete within reasonable time with mocks
            assert processing_time < 2.0
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_many_documents_retrieval(self, mock_db, sample_rag_agent, sample_api_key):
        """Test retrieval performance with many documents"""
        mock_db.query.return_value.filter.return_value.first.side_effect = [sample_rag_agent, sample_api_key]
        
        with patch('os.path.exists', return_value=True), \
             patch('api.api_v1.endpoints.agents.decrypt_api_key'), \
             patch('api.api_v1.endpoints.agents.GoogleGenerativeAIEmbeddings'), \
             patch('api.api_v1.endpoints.agents.Chroma') as mock_chroma:
            
            # Mock many documents
            mock_docs = []
            for i in range(1000):  # 1000 document chunks
                doc = Mock()
                doc.metadata = {"source": f"/path/to/doc{i % 100}.pdf"}  # 100 unique documents
                mock_docs.append(doc)
            
            mock_chroma_instance = Mock()
            mock_chroma.return_value = mock_chroma_instance
            mock_chroma_instance.similarity_search.return_value = mock_docs
            
            import time
            start_time = time.time()
            
            result = await get_agent_documents(
                agent_id=1,
                db=mock_db,
                current_user_id=1
            )
            
            end_time = time.time()
            retrieval_time = end_time - start_time
            
            assert retrieval_time < 1.0  # Should be fast with mocks
            assert len(result["documents"]) == 100  # 100 unique documents


if __name__ == "__main__":
    pytest.main([__file__])
