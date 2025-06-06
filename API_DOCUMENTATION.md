# AI Superapp Backend API Documentation

## Overview
This is a FastAPI-based backend for an AI Superapp that allows users to create and run custom AI workflows called "mini services" using various AI agents.

## Base URL
`http://127.0.0.1:8000` (Development)

## Authentication
Most endpoints require user authentication. Users need to provide `current_user_id` as a parameter (this should be replaced with proper JWT authentication in production).

---

## Authentication Endpoints

### POST `/api/v1/auth/login`
**Description**: Authenticate user and receive JWT token

**Request Body**:
```json
{
  "username": "user@example.com",
  "password": "password123"
}
```

**Response**:
```json
{
  "access_token": "jwt_token_here",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "username": "testuser",
    "email": "user@example.com"
  }
}
```

### POST `/api/v1/auth/register`
**Description**: Register a new user account

**Request Body**:
```json
{
  "username": "newuser",
  "email": "newuser@example.com",
  "password": "password123"
}
```

---

## Mini Services Endpoints

Mini Services are custom AI workflows that chain together multiple AI agents to accomplish complex tasks.

### POST `/api/v1/mini-services/`
**Description**: Create a new mini service

**Parameters**:
- `current_user_id` (required): ID of the authenticated user

**Request Body**:
```json
{
  "name": "Text Summarizer",
  "description": "Summarizes long text using AI",
  "workflow": {
    "nodes": {
      "0": {"agent_id": 1, "next": 1},
      "1": {"agent_id": 2, "next": null}
    }
  },
  "input_type": "text",
  "output_type": "text",
  "is_public": false
}
```

**Response**: Returns the created mini service with ID and metadata.

### GET `/api/v1/mini-services/`
**Description**: List all mini services accessible to the current user (owned + public)

**Parameters**:
- `current_user_id` (required): ID of the authenticated user
- `skip` (optional): Number of records to skip for pagination (default: 0)
- `limit` (optional): Maximum number of records to return (default: 100)

**Response**: Array of mini service objects with pricing information and owner details.

### GET `/api/v1/mini-services/{service_id}`
**Description**: Get details of a specific mini service

**Parameters**:
- `service_id` (path): ID of the mini service
- `current_user_id` (required): ID of the authenticated user

### POST `/api/v1/mini-services/{service_id}/run`
**Description**: Execute a mini service with input data

**Parameters**:
- `service_id` (path): ID of the mini service to run
- `current_user_id` (required): ID of the authenticated user

**Request Body**:
```json
{
  "input": "Your input text here",
  "context": {"optional": "context data"},
  "api_keys": {
    "1": "sk-openai-api-key",
    "2": "gemini-api-key"
  }
}
```

**Response**:
```json
{
  "output": "Final result from the workflow",
  "token_usage": {
    "total_tokens": 150,
    "pricing": {"cost": 0.0023, "currency": "USD"}
  },
  "process_id": 123,
  "audio_urls": [],
  "results": []
}
```

### PUT `/api/v1/mini-services/{service_id}`
**Description**: Update an existing mini service

**Parameters**:
- `service_id` (path): ID of the mini service
- `current_user_id` (required): ID of the authenticated user

### DELETE `/api/v1/mini-services/{service_id}`
**Description**: Delete a mini service and its related processes

**Parameters**:
- `service_id` (path): ID of the mini service
- `current_user_id` (required): ID of the authenticated user

**Response**: 204 No Content on success

### POST `/api/v1/mini-services/upload`
**Description**: Upload a file for processing by mini services

**Parameters**:
- `current_user_id` (required): ID of the authenticated user

**Request**: Multipart form data with file

**Limits**: Maximum file size is 200MB

**Response**:
```json
{
  "filename": "unique_filename.ext",
  "file_path": "_INPUT/unique_filename.ext",
  "file_size": 1024000
}
```

### POST `/api/v1/mini-services/chat-generate`
**Description**: Interactive chat interface for creating mini services using AI assistance

**Parameters**:
- `current_user_id` (required): ID of the authenticated user

**Request Body**:
```json
{
  "message": "I want to create a service that translates text to Spanish",
  "conversation_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ],
  "approve_service": false,
  "service_specification": {},
  "gemini_api_key": "optional-api-key"
}
```

**Response Types**:

1. **Chat Response** (gathering requirements):
```json
{
  "type": "chat_response",
  "message": "What type of input will your service accept?",
  "conversation_history": [...],
  "checklist": {...},
  "workflow_state": {...}
}
```

2. **Approval Required** (specifications ready):
```json
{
  "type": "approval_required",
  "message": "Review your service specifications",
  "service_specification": {...},
  "conversation_history": [...],
  "checklist": {...}
}
```

3. **Service Created** (final step):
```json
{
  "type": "service_created",
  "message": "Your service has been created!",
  "mini_service": {...},
  "agents": [...],
  "conversation_history": [...]
}
```

### GET `/api/v1/mini-services/{service_id}/audio-files`
**Description**: List all audio files generated by a specific mini service

**Parameters**:
- `service_id` (path): ID of the mini service
- `current_user_id` (required): ID of the authenticated user

---

## Process Endpoints

Processes represent execution instances of mini services.

### GET `/api/v1/processes/`
**Description**: List recent processes for the current user

**Parameters**:
- `current_user_id` (required): ID of the authenticated user
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Maximum number of records to return (default: 10)

**Response**: Array of process objects with mini service details and pricing information.

### GET `/api/v1/processes/recent-activities`
**Description**: Get recent activity logs for the current user

**Parameters**:
- `current_user_id` (required): ID of the authenticated user
- `limit` (optional): Maximum number of logs to return (default: 5)

**Response**: Array of log entries with activity details.

---

## Agent Endpoints

Agents are individual AI components that can be chained together in mini services.

### POST `/api/v1/agents/`
**Description**: Create a new AI agent

**Request Body**:
```json
{
  "name": "OpenAI GPT Agent",
  "agent_type": "openai",
  "system_instruction": "You are a helpful assistant",
  "config": {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
  },
  "input_type": "text",
  "output_type": "text"
}
```

### GET `/api/v1/agents/`
**Description**: List all agents accessible to the current user

### GET `/api/v1/agents/{agent_id}`
**Description**: Get details of a specific agent

### PUT `/api/v1/agents/{agent_id}`
**Description**: Update an existing agent

### DELETE `/api/v1/agents/{agent_id}`
**Description**: Delete an agent

---

## API Keys Endpoints

Manage API keys for various AI services.

### POST `/api/v1/api-keys/`
**Description**: Add a new API key for an AI service

**Request Body**:
```json
{
  "provider": "openai",
  "api_key": "sk-your-api-key-here",
  "description": "OpenAI API key for GPT models"
}
```

### GET `/api/v1/api-keys/`
**Description**: List all API keys for the current user (keys are masked for security)

### DELETE `/api/v1/api-keys/{key_id}`
**Description**: Delete an API key

---

## Favorites Endpoints

Manage favorite mini services.

### POST `/api/v1/favorites/`
**Description**: Add a mini service to favorites

**Request Body**:
```json
{
  "mini_service_id": 123
}
```

### GET `/api/v1/favorites/`
**Description**: List favorite mini services

### DELETE `/api/v1/favorites/{mini_service_id}`
**Description**: Remove a mini service from favorites

### GET `/api/v1/favorites/counts`
**Description**: Get favorite counts for mini services

---

## Available Agent Types

The system supports various types of AI agents:

1. **openai** - OpenAI GPT models (text → text)
2. **gemini** - Google Gemini models (text → text)
3. **edge_tts** - Microsoft Edge Text-to-Speech (text → audio)
4. **bark_tts** - Bark high-quality TTS (text → audio)
5. **transcribe** - Speech-to-text transcription (audio → text)
6. **gemini_text2image** - Image generation (text → image)
7. **internet_research** - Web search (text → text)
8. **document_parser** - Document text extraction (document → text)
9. **google_translate** - Language translation (text → text)
10. **rag** - Retrieval Augmented Generation (text → text)
11. **custom_endpoint_llm** - Custom API endpoint (text → text)

## Error Responses

All endpoints return appropriate HTTP status codes:

- `200` - Success
- `201` - Created
- `204` - No Content (for deletions)
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (missing authentication)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `413` - Request Entity Too Large (file too big)
- `422` - Unprocessable Entity (validation errors)
- `500` - Internal Server Error

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## File Storage

- **Input files**: Stored in `_INPUT/` directory
- **Output files**: Stored in `_OUTPUT/` directory
- **Static files**: Served from `/_OUTPUT` endpoint

## Rate Limits and Quotas

Currently no rate limits are implemented, but this should be added for production use.

## Security Considerations

1. API keys are encrypted before storage
2. File uploads are limited to 200MB
3. User can only access their own private services
4. Public services are accessible to all users
5. SQL injection protection through ORM usage
6. Input validation on all endpoints

## WebSocket Support

Currently not implemented, but would be useful for real-time updates during mini service execution.
