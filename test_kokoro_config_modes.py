"""
Test script to demonstrate Kokoro TTS Agent configuration modes

This demonstrates the difference between:
1. Single text mode (podcast_mode: False)
2. Podcast mode (podcast_mode: True)
"""

import asyncio
import json
from agents.kokoro_agent import KokoroTTSAgent

async def test_single_text_mode():
    """Test single text mode with podcast_mode disabled"""
    
    print("=== Testing Single Text Mode ===")
    
    # Configuration for single text mode
    config = {
        "voice": "af_sarah",
        "speed": 1.2,
        "language": "en-us",
        "podcast_mode": False,  # Single text mode
        "model_path": "./model_files/kokkoro-82m/kokoro-v1.0.onnx",
        "voices_path": "./model_files/kokkoro-82m/voices/voices-v1.0.bin"
    }
    
    agent = KokoroTTSAgent(config, "Single text TTS agent")
    
    # Test with simple string input
    print("Test 1: Simple string input")
    text = "Hello world! This is a test of the Kokoro TTS system in single text mode."
    result = await agent.process(text, {"filename": "single_text_test.wav"})
    print(f"Output: {result.get('output')}")
    print(f"Voice used: {result.get('voice')}")
    print(f"Mode: Single text")
    print()
    
    # Test with dict input (like from RAG)
    print("Test 2: Dictionary input (RAG-style)")
    rag_response = {
        "response": "This is a response from a RAG system that needs to be converted to speech.",
        "sources": ["doc1.pdf", "doc2.txt"]
    }
    result = await agent.process(rag_response, {"filename": "rag_response_test.wav"})
    print(f"Output: {result.get('output')}")
    print(f"Voice used: {result.get('voice')}")
    print()

async def test_podcast_mode():
    """Test podcast mode with podcast_mode enabled"""
    
    print("=== Testing Podcast Mode ===")
    
    # Configuration for podcast mode
    config = {
        "voice": "af_sarah",  # Default voice (not used in podcast mode)
        "speed": 1.0,
        "language": "en-us",
        "podcast_mode": True,  # Enable podcast mode
        "model_path": "./model_files/kokkoro-82m/kokoro-v1.0.onnx",
        "voices_path": "./model_files/kokkoro-82m/voices/voices-v1.0.bin"
    }
    
    agent = KokoroTTSAgent(config, "Podcast TTS agent")
    
    # Test with proper podcast format
    print("Test 1: Proper podcast format")
    podcast_data = {
        "sentences": [
            {"text": "Welcome to our technology podcast!", "voice": "af_sarah"},
            {"text": "Today we're discussing artificial intelligence.", "voice": "am_michael"},
            {"text": "It's a fascinating topic with many applications.", "voice": "af_nicole"}
        ]
    }
    
    context = {
        "filename": "config_podcast_test.wav",
        "min_pause_duration": 0.5,
        "max_pause_duration": 1.2
    }
    
    result = await agent.process(podcast_data, context)
    print(f"Output: {result.get('output')}")
    print(f"Voices used: {result.get('voices_used')}")
    print(f"Sentence count: {result.get('sentence_count')}")
    print(f"Mode: {result.get('voice_info', {}).get('mode', 'unknown')}")
    print()
    
    # Test with direct list format
    print("Test 2: Direct list format")
    podcast_list = [
        {"text": "This is the first sentence.", "voice": "bf_emma"},
        {"text": "And this is the second sentence.", "voice": "bm_george"}
    ]
    
    result = await agent.process(podcast_list, {"filename": "list_format_test.wav"})
    print(f"Output: {result.get('output')}")
    print(f"Voices used: {result.get('voices_used')}")
    print()

async def test_mode_validation():
    """Test validation when wrong input format is used for each mode"""
    
    print("=== Testing Mode Validation ===")
    
    # Single text mode agent
    single_config = {"podcast_mode": False, "voice": "af_sarah", "speed": 1.0, "language": "en-us"}
    single_agent = KokoroTTSAgent(single_config, "Single text agent")
    
    # Podcast mode agent  
    podcast_config = {"podcast_mode": True, "voice": "af_sarah", "speed": 1.0, "language": "en-us"}
    podcast_agent = KokoroTTSAgent(podcast_config, "Podcast agent")
    
    print("Test 1: Single text mode works with any input")
    # Single text mode should work with various inputs
    inputs_for_single = [
        "Simple string",
        {"response": "Dictionary with response"},
        {"sentences": [{"text": "Even podcast format", "voice": "af_sarah"}]}  # Will convert to string
    ]
    
    for i, input_data in enumerate(inputs_for_single):
        result = await single_agent.process(input_data, {"filename": f"single_test_{i}.wav"})
        print(f"  Input {i+1}: {type(input_data).__name__} -> {result.get('agent_type')} (success: {'error' not in result})")
    print()
    
    print("Test 2: Podcast mode requires specific format")
    # Podcast mode should validate input format
    inputs_for_podcast = [
        "Simple string",  # Should fail
        {"response": "Wrong dict format"},  # Should fail
        {"sentences": [{"text": "Correct format", "voice": "af_sarah"}]},  # Should succeed
        [{"text": "List format", "voice": "af_sarah"}]  # Should succeed
    ]
    
    for i, input_data in enumerate(inputs_for_podcast):
        result = await podcast_agent.process(input_data, {"filename": f"podcast_test_{i}.wav"})
        success = 'error' not in result
        print(f"  Input {i+1}: {type(input_data).__name__} -> {'Success' if success else 'Validation Error'}")
        if not success and 'format_requirements' in result:
            print(f"    Error: {result.get('error', 'Unknown error')[:100]}...")

async def demo_configuration_comparison():
    """Demo showing the configuration differences"""
    
    print("=== Configuration Comparison Demo ===")
    
    print("Single Text Mode Configuration:")
    single_config = KokoroTTSAgent.get_default_config()
    single_config["podcast_mode"] = False
    print(json.dumps(single_config, indent=2))
    print()
    
    print("Podcast Mode Configuration:")
    podcast_config = KokoroTTSAgent.get_default_config()
    podcast_config["podcast_mode"] = True
    print(json.dumps(podcast_config, indent=2))
    print()
    
    print("Key Differences:")
    print("- podcast_mode: False = Single text mode (accepts any text input)")
    print("- podcast_mode: True = Podcast mode (requires sentences with voice assignments)")
    print("- Context parameters like min_pause_duration only apply to podcast mode")
    print("- Single text mode uses the 'voice' config parameter")
    print("- Podcast mode uses individual voice assignments per sentence")

if __name__ == "__main__":
    print("Kokoro TTS Agent - Configuration Mode Testing")
    print("=" * 60)
    
    asyncio.run(demo_configuration_comparison())
    print("\n" + "=" * 60 + "\n")
    
    asyncio.run(test_single_text_mode())
    print("\n" + "=" * 60 + "\n")
    
    asyncio.run(test_podcast_mode())
    print("\n" + "=" * 60 + "\n")
    
    asyncio.run(test_mode_validation())
