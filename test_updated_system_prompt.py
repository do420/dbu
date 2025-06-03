#!/usr/bin/env python3

"""
Test script to verify the updated system prompt with agent type details.
"""

import json

def test_updated_system_prompt():
    """Test that the system prompt includes correct agent type information"""
    
    # Read the mini_services.py file
    with open('/Users/denizozcan/Desktop/app/api/api_v1/endpoints/mini_services.py', 'r') as f:
        content = f.read()
    
    print("✅ Testing updated system prompt...")
    
    # Check if the agent types include input/output information
    agent_types_to_check = [
        "gemini: AI text generation and analysis (text → text)",
        "edge_tts: Convert text to speech (text → sound)",
        "bark_tts: High-quality text to speech (text → sound)",
        "transcribe: Convert audio to text (sound → text)",
        "gemini_text2image: Create images from descriptions (text → image)",
        "document_parser: Extract text from documents (document → text)",
        "google_translate: Translate between languages (text → text)"
    ]
    
    for agent_type in agent_types_to_check:
        if agent_type in content:
            print(f"✅ Found: {agent_type}")
        else:
            print(f"❌ Missing: {agent_type}")
    
    # Check if outdated agent types are removed
    outdated_types = [
        "local: Local AI models",
        "tts: Text-to-speech conversion",
        "custom_endpoint: Call custom API endpoints",
        "gemini_image_generation: Generate images using Gemini"
    ]
    
    for outdated_type in outdated_types:
        if outdated_type in content:
            print(f"❌ Still contains outdated: {outdated_type}")
        else:
            print(f"✅ Correctly removed: {outdated_type}")
    
    # Check if the conversation style is updated
    conversation_style_checks = [
        "Be conversational and friendly",
        "Ask ONE question at a time",
        "Keep responses short (2-3 sentences max)",
        "Avoid technical jargon",
        "Automatically suggest and select appropriate tools"
    ]
    
    for check in conversation_style_checks:
        if check in content:
            print(f"✅ Found conversation style rule: {check}")
        else:
            print(f"❌ Missing conversation style rule: {check}")
    
    # Check if the agent type list in JSON format is correct
    correct_agent_types = [
        "gemini|openai|claude|edge_tts|bark_tts|transcribe|gemini_text2image|internet_research|document_parser|custom_endpoint_llm|rag|google_translate"
    ]
    
    for agent_type_list in correct_agent_types:
        if agent_type_list in content:
            print(f"✅ Found correct agent type list in JSON format")
        else:
            print(f"❌ Agent type list in JSON format needs update")
    
    print("\n✅ System prompt update test completed!")

if __name__ == "__main__":
    test_updated_system_prompt()
