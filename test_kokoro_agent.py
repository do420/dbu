"""
Test file for Kokoro TTS Agent

This file demonstrates how to use the Kokoro TTS agent in the project.
Make sure you have:
1. Installed kokoro-onnx: pip install -U kokoro-onnx soundfile
2. Downloaded the model files:
   - kokoro-v1.0.onnx
   - voices-v1.0.bin
3. Placed them in ./model_files/kokkoro-82m/ directory
"""

import asyncio
import json
from agents.kokoro_agent import KokoroTTSAgent

async def test_kokoro_agent():
    """Test the Kokoro TTS agent with various configurations"""
    
    # Test configuration
    config = {
        "voice": "af_sarah",
        "speed": 1.0,
        "language": "en-us",
        "model_path": "./model_files/kokkoro-82m/kokoro-v1.0.onnx",
        "voices_path": "./model_files/kokkoro-82m/voices/voices-v1.0.bin"
    }
    
    # Create agent instance
    agent = KokoroTTSAgent(config, "You are a text-to-speech agent using Kokoro TTS.")
    
    # Test 1: Basic text conversion
    print("Test 1: Basic text conversion")
    test_text = "Hello and welcome to the Kokoro TTS demonstration! This is a test of neural text-to-speech synthesis."
    
    result1 = await agent.process(test_text)
    print(f"Result: {result1}")
    print(f"Audio file created: {result1.get('audio_file')}")
    print()
    
    # Test 2: Different voice
    print("Test 2: Different voice (am_michael)")
    context = {"voice": "am_michael"}
    
    result2 = await agent.process("This is a test with a different voice - Michael speaking!", context)
    print(f"Voice used: {result2.get('voice')}")
    print(f"Audio file created: {result2.get('audio_file')}")
    print()
    
    # Test 3: Speed variation
    print("Test 3: Speed variation")
    context = {"voice": "af_nicole", "speed": 1.2}
    
    result3 = await agent.process("Testing speed variation with Kokoro TTS. This should sound a bit faster.", context)
    print(f"Speed used: {result3.get('voice_info', {}).get('speed')}")
    print(f"Audio file created: {result3.get('audio_file')}")
    print()
    
    # Test 4: Different language
    print("Test 4: Different language (es - Spanish)")
    context = {"language": "es", "voice": "af_sarah"}
    
    result4 = await agent.process("Hola, esto es una prueba del sistema Kokoro TTS en español.", context)
    print(f"Language used: {result4.get('voice_info', {}).get('language')}")
    print(f"Audio file created: {result4.get('audio_file')}")
    print()
    
    # Display available voices and languages
    print("Available voices:", KokoroTTSAgent.get_available_voices())
    print("Supported languages:", KokoroTTSAgent.get_supported_languages())
    print("Default config:", KokoroTTSAgent.get_default_config())

if __name__ == "__main__":
    asyncio.run(test_kokoro_agent())
