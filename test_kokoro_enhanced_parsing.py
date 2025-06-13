"""
Test script to verify the enhanced markdown JSON parsing functionality

This tests the specific case that was failing:
'```json\n{\n  "sentences": [\n    {\n      "voice": "af_sarah",\n      "text": "Dogs are truly man\'s best friend..."\n    }\n  ]\n}\n```\n'
"""

import asyncio
import json
import logging
from agents.kokoro_agent import KokoroTTSAgent

# Set up logging to see the debug output
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

async def test_problematic_markdown_input():
    """Test the specific input format that was failing"""
    
    print("=== Testing Enhanced Markdown JSON Parsing ===")
    
    # Configuration with podcast mode enabled
    config = {
        "voice": "af_sarah",
        "speed": 1.0,
        "language": "en-us",
        "podcast_mode": True,  # Enable podcast mode
        "model_path": "./model_files/kokkoro-82m/kokoro-v1.0.onnx",
        "voices_path": "./model_files/kokkoro-82m/voices/voices-v1.0.bin"
    }
    
    agent = KokoroTTSAgent(config, "Enhanced markdown parsing test agent")
    
    # Test 1: The exact input that was failing
    print("Test 1: The problematic input format")
    problematic_input = '```json\n{\n  "sentences": [\n    {\n      "voice": "af_sarah",\n      "text": "Dogs are truly man\'s best friend, aren\'t they? Their loyalty and companionship are unparalleled."\n    },\n    {\n      "voice": "am_michael",\n      "text": "Absolutely!  And their boundless energy and playful nature always bring a smile to my face."\n    }\n  ]\n}\n```\n'
    
    print(f"Input string representation: {repr(problematic_input)}")
    print(f"Input length: {len(problematic_input)}")
    print(f"Starts with: {repr(problematic_input[:20])}")
    print(f"Ends with: {repr(problematic_input[-20:])}")
    
    result = await agent.process(problematic_input, {"filename": "problematic_test.wav"})
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        if 'format_requirements' in result:
            print("Format requirements:")
            print(result['format_requirements']['example'])
    else:
        print(f"✅ Success! Generated audio file: {result.get('audio_file')}")
        print(f"Voices used: {result.get('voices_used')}")
        print(f"Sentence count: {result.get('sentence_count')}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Various other markdown formats
    test_cases = [
        # Case 1: Standard markdown with exact newlines
        {
            "name": "Standard markdown code block",
            "input": "```json\n{\n  \"sentences\": [\n    {\"text\": \"Hello world!\", \"voice\": \"af_sarah\"}\n  ]\n}\n```"
        },
        
        # Case 2: Markdown with extra trailing newline
        {
            "name": "Markdown with trailing newline",
            "input": "```json\n{\n  \"sentences\": [\n    {\"text\": \"Hello world!\", \"voice\": \"af_sarah\"}\n  ]\n}\n```\n"
        },
        
        # Case 3: Simple json prefix
        {
            "name": "Simple json prefix",
            "input": "json\n{\n  \"sentences\": [\n    {\"text\": \"Hello world!\", \"voice\": \"af_sarah\"}\n  ]\n}"
        },
        
        # Case 4: Direct JSON string
        {
            "name": "Direct JSON string", 
            "input": "{\"sentences\": [{\"text\": \"Hello world!\", \"voice\": \"af_sarah\"}]}"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+2}: {test_case['name']}")
        print(f"Input: {repr(test_case['input'][:50])}...")
        
        result = await agent.process(test_case['input'], {"filename": f"test_{i+2}.wav"})
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success! Generated audio file: {result.get('audio_file')}")
        
        print()

async def test_error_cases():
    """Test various error cases to ensure good error reporting"""
    
    print("=== Testing Error Cases with Detailed Debug ===")
    
    config = {
        "podcast_mode": True,
        "voice": "af_sarah",
        "speed": 1.0,
        "language": "en-us"
    }
    
    agent = KokoroTTSAgent(config, "Error testing agent")
    
    error_cases = [
        {
            "name": "Invalid JSON in markdown",
            "input": "```json\n{invalid json}\n```"
        },
        {
            "name": "Missing closing backticks",
            "input": "```json\n{\"sentences\": []}"
        },
        {
            "name": "Empty sentences",
            "input": "```json\n{\"sentences\": []}\n```"
        },
        {
            "name": "Missing voice field",
            "input": "```json\n{\"sentences\": [{\"text\": \"Hello\"}]}\n```"
        },
        {
            "name": "Invalid voice",
            "input": "```json\n{\"sentences\": [{\"text\": \"Hello\", \"voice\": \"invalid_voice\"}]}\n```"
        }
    ]
    
    for i, test_case in enumerate(error_cases):
        print(f"Error Test {i+1}: {test_case['name']}")
        
        result = await agent.process(test_case['input'], {})
        
        if "error" in result:
            print(f"✅ Expected error: {result['error'][:100]}...")
            if 'format_requirements' in result:
                print("📝 Format requirements provided ✓")
        else:
            print(f"❌ Unexpected success: {result}")
        
        print()

if __name__ == "__main__":
    print("Testing Enhanced Kokoro TTS Markdown JSON Parsing...")
    print("=" * 60)
    
    asyncio.run(test_problematic_markdown_input())
    asyncio.run(test_error_cases())
