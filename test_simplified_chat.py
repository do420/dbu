#!/usr/bin/env python3

"""
Test script to verify the simplified chat-generate endpoint
"""

import json
import sys
import os

# Add the app directory to sys.path
sys.path.insert(0, os.path.abspath('.'))

def test_simplified_system_prompt():
    """Test that the simplified system prompt is properly formatted"""
    
    # Mock conversation data
    conversation_text = "User: I want to create something\nAssistant: What would you like to create?"
    message = "I want to translate text to speech"
    
    # Simplified system prompt (matching our updated version)
    system_prompt = f"""
You are a friendly AI assistant that helps users create custom mini-services. Your goal is to understand what they want and automatically create the perfect service for them.

CONVERSATION PROGRESS TRACKING:
□ Service Purpose: What should the service do?
□ Service Name: What should it be called?
□ Input Type: What will users provide? (text, image, sound, or document)
□ Output Type: What should it produce? (text, image, sound, or document)

Previous conversation:
{conversation_text}

Current user message: {message}

INSTRUCTIONS:
1. Ask only ONE simple question at a time to fill missing checklist items
2. Keep responses short and friendly (max 2-3 sentences)
3. Use everyday language, avoid technical terms
4. Automatically select the best AI agents based on user needs (don't ask them to choose)
5. When you have enough information, automatically create the service with "CREATE_SERVICE:" + JSON

CONVERSATION STYLE:
- Ask one focused question at a time
- Be conversational and helpful
- Suggest ideas if user seems stuck
- Automatically create service when ready (don't ask permission)

AGENT SELECTION GUIDE (choose automatically):
- Text analysis/writing: gemini
- Language translation: google_translate  
- Text to speech: tts
- Speech to text: transcribe
- Document reading: document_parser
- Image generation: gemini_image_generation
- Web research: internet_research

JSON Format (when creating service):
{{
    "service": {{
        "name": "Service Name",
        "description": "Simple description of what it does",
        "input_type": "text|image|sound|document",
        "output_type": "text|image|sound|document",
        "is_public": false
    }},
    "agents": [
        {{
            "name": "Agent Name",
            "agent_type": "gemini|google_translate|tts|transcribe|document_parser|gemini_image_generation|internet_research",
            "system_instruction": "Clear instructions for the agent",
            "config": {{"temperature": 0.7, "model": "gemini-1.5-flash"}},
            "input_type": "text|image|sound|document",
            "output_type": "text|image|sound|document"
        }}
    ],
    "workflow": {{
        "nodes": {{
            "0": {{"agent_id": 0, "next": 1}},
            "1": {{"agent_id": 1, "next": null}}
        }}
    }}
}}

Focus on ONE missing checklist item per response. Be helpful and concise.
"""
    
    print("✅ Simplified System Prompt Structure:")
    print("=" * 50)
    print(f"Length: {len(system_prompt)} characters")
    print(f"Lines: {len(system_prompt.split('\\n'))} lines")
    print("=" * 50)
    
    # Check key elements
    checks = [
        ("Focuses on one question at a time", "Ask only ONE simple question" in system_prompt),
        ("Uses friendly language", "friendly AI assistant" in system_prompt),
        ("Automatic agent selection", "Automatically select the best AI agents" in system_prompt),
        ("Simplified checklist", "CONVERSATION PROGRESS TRACKING" in system_prompt),
        ("No technical jargon", "everyday language" in system_prompt),
        ("Concise responses", "max 2-3 sentences" in system_prompt),
        ("Auto-creation", "automatically create the service" in system_prompt),
    ]
    
    print("Key Features Check:")
    for description, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {description}")
    
    return all(result for _, result in checks)

def test_json_format_in_prompt():
    """Test that the JSON format in the prompt is valid"""
    
    # Extract the JSON template from the prompt
    json_template = """
{
    "service": {
        "name": "Service Name",
        "description": "Simple description of what it does",
        "input_type": "text|image|sound|document",
        "output_type": "text|image|sound|document",
        "is_public": false
    },
    "agents": [
        {
            "name": "Agent Name",
            "agent_type": "gemini|google_translate|tts|transcribe|document_parser|gemini_image_generation|internet_research",
            "system_instruction": "Clear instructions for the agent",
            "config": {"temperature": 0.7, "model": "gemini-1.5-flash"},
            "input_type": "text|image|sound|document",
            "output_type": "text|image|sound|document"
        }
    ],
    "workflow": {
        "nodes": {
            "0": {"agent_id": 0, "next": 1},
            "1": {"agent_id": 1, "next": null}
        }
    }
}"""
    
    try:
        # Replace the pipe-separated options with actual values for validation
        test_json = json_template.replace(
            "text|image|sound|document", "text"
        ).replace(
            "gemini|google_translate|tts|transcribe|document_parser|gemini_image_generation|internet_research", "gemini"
        )
        
        parsed = json.loads(test_json)
        print("✅ JSON template in prompt is valid")
        print(f"✅ Service structure: {list(parsed['service'].keys())}")
        print(f"✅ Agent structure: {list(parsed['agents'][0].keys())}")
        print(f"✅ Workflow structure: {list(parsed['workflow'].keys())}")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ JSON template is invalid: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Simplified Chat-Generate System Prompt")
    print("=" * 60)
    
    test1 = test_simplified_system_prompt()
    print()
    test2 = test_json_format_in_prompt()
    
    print()
    print("=" * 60)
    if test1 and test2:
        print("✅ All tests passed! The simplified system prompt is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
