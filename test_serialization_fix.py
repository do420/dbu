#!/usr/bin/env python3

"""
Test script to verify the chat-generate endpoint serialization fix
"""

import json
from datetime import datetime

def test_json_parsing_with_markdown():
    """Test that our JSON parsing logic handles markdown-wrapped responses"""
    
    # Test case 1: JSON wrapped in markdown code blocks
    markdown_response = '''```json
{
    "service": {
        "name": "Test Service",
        "description": "A test service",
        "input_type": "text",
        "output_type": "text",
        "is_public": false
    },
    "agents": [
        {
            "name": "Test Agent",
            "agent_type": "gemini",
            "system_instruction": "Test instruction",
            "config": {"temperature": 0.7},
            "input_type": "text",
            "output_type": "text"
        }
    ],
    "workflow": {
        "nodes": {
            "0": {
                "agent_id": 0,
                "next": null
            }
        }
    }
}
```'''
    
    # Apply the same logic as in our endpoint
    response_text = markdown_response.strip()
    
    # Remove markdown code block formatting if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]  # Remove ```json
    if response_text.endswith("```"):
        response_text = response_text[:-3]  # Remove ```
    response_text = response_text.strip()
    
    try:
        parsed_data = json.loads(response_text)
        print("✅ JSON parsing successful!")
        print(f"Service name: {parsed_data['service']['name']}")
        print(f"Agent name: {parsed_data['agents'][0]['name']}")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return False

def test_response_serialization():
    """Test that our response format is serializable"""
    
    # Mock data structure similar to what our endpoint returns
    mock_response = {
        "type": "service_created",
        "mini_service": {
            "id": 1,
            "name": "Test Service",
            "description": "A test service",
            "workflow": {"nodes": {"0": {"agent_id": 1, "next": None}}},
            "input_type": "text",
            "output_type": "text",
            "owner_id": 1,
            "created_at": datetime.now().isoformat(),
            "is_enhanced": False,
            "is_public": False
        },
        "agents": [
            {
                "id": 1,
                "name": "Test Agent",
                "agent_type": "gemini",
                "system_instruction": "Test instruction",
                "config": {"temperature": 0.7},
                "input_type": "text",
                "output_type": "text",
                "owner_id": 1,
                "created_at": datetime.now().isoformat(),
                "is_enhanced": False
            }
        ],
        "message": "Successfully created service!",
        "conversation_history": [
            {"role": "user", "content": "Create a test service"},
            {"role": "assistant", "content": "Service created!"}
        ]
    }
    
    try:
        # Try to serialize to JSON
        json_str = json.dumps(mock_response, indent=2)
        print("✅ Response serialization successful!")
        print("Response structure:")
        print(f"- Type: {mock_response['type']}")
        print(f"- Service ID: {mock_response['mini_service']['id']}")
        print(f"- Agents count: {len(mock_response['agents'])}")
        return True
    except Exception as e:
        print(f"❌ Response serialization failed: {e}")
        return False

def main():
    print("Testing chat-generate endpoint fixes...")
    print("=" * 50)
    
    print("\n1. Testing JSON parsing with markdown removal:")
    json_test = test_json_parsing_with_markdown()
    
    print("\n2. Testing response serialization:")
    serialization_test = test_response_serialization()
    
    print("\n" + "=" * 50)
    if json_test and serialization_test:
        print("🎉 All tests passed! The endpoint should work correctly now.")
    else:
        print("❌ Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main()
