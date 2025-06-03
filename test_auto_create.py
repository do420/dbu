#!/usr/bin/env python3

"""
Test script to verify automatic service creation detection
"""

import requests
import json

# Test the chat-generate endpoint
def test_auto_create():
    url = "http://localhost:8000/api/v1/mini-services/chat-generate"
    
    # Headers (you'll need to replace with actual auth token)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_TOKEN_HERE"  # Replace with actual token
    }
    
    # Test message that should trigger automatic creation
    payload = {
        "message": "Create a joke generator service that takes text input and returns funny jokes",
        "conversation_history": [
            {"role": "user", "content": "Hi, I want to create a service"},
            {"role": "assistant", "content": "Great! What would you like your service to do?"},
            {"role": "user", "content": "I want it to generate jokes"},
            {"role": "assistant", "content": "Perfect! What type of input will you provide? (text, image, or audio)"},
            {"role": "user", "content": "Text input"},
            {"role": "assistant", "content": "Got it! And what would you like to get back?"},
            {"role": "user", "content": "Text with jokes"},
            {"role": "assistant", "content": "Excellent! What would you like to name this service?"}
        ]
    }
    
    print("Testing automatic service creation...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("type") == "service_created":
                print("✅ SUCCESS: Service was automatically created!")
            elif result.get("type") == "chat_response":
                print("⚠️  ISSUE: Got chat response instead of automatic creation")
                print(f"Message: {result.get('message')}")
            else:
                print(f"❓ UNKNOWN: Unexpected response type: {result.get('type')}")
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_auto_create()
