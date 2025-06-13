"""
Test script for Kokoro TTS Agent Podcast Feature

This demonstrates how to use the podcast mode of the Kokoro TTS agent.
"""

import asyncio
import json
from agents.kokoro_agent import KokoroTTSAgent

async def test_podcast_mode():
    """Test the Kokoro TTS agent podcast mode"""
    
    # Test configuration with podcast_mode enabled in config (not context)
    config = {
        "voice": "af_sarah",  # Default voice
        "speed": 1.0,
        "language": "en-us",
        "podcast_mode": True,  # Now configured in agent config
        "model_path": "./model_files/kokkoro-82m/kokoro-v1.0.onnx",
        "voices_path": "./model_files/kokkoro-82m/voices/voices-v1.0.bin"
    }
    
    # Create agent instance with podcast mode enabled
    agent = KokoroTTSAgent(config, "You are a podcast TTS agent using Kokoro TTS.")
    
    # Test podcast sentences (similar to testing-kokorov2.py)
    podcast_sentences = [
        {"voice": "af_sarah", "text": "Hello and welcome to the podcast! We've got some exciting things lined up today."},
        {"voice": "am_michael", "text": "It's going to be an exciting episode. Stick with us!"},
        {"voice": "af_sarah", "text": "But first, we've got a special guest with us. Please welcome Nicole!"},
        {"voice": "af_sarah", "text": "Now, we've been told Nicole has a very unique way of speaking today... a bit of a mysterious vibe, if you will."},
        {"voice": "af_nicole", "text": "Hey there... I'm so excited to be a guest today... But I thought I'd keep it quiet... for now..."},
        {"voice": "am_michael", "text": "Well, it certainly adds some intrigue! Let's dive in and see what that's all about."},
        {"voice": "af_sarah", "text": "Today, we're covering something that's close to our hearts"},
        {"voice": "am_michael", "text": "It's going to be a good one!"}
    ]
    
    # Test 1: Basic podcast with list of sentences
    print("Test 1: Podcast mode with list input")
    context = {
        "min_pause_duration": 0.3,
        "max_pause_duration": 1.0,
        "speed": 1.0
    }
    
    result1 = await agent.process(podcast_sentences, context)
    print(f"Result: {json.dumps(result1, indent=2)}")
    print(f"Podcast audio file created: {result1.get('audio_file')}")
    print(f"Voices used: {result1.get('voices_used')}")
    print(f"Sentence count: {result1.get('sentence_count')}")
    print()
    
    # Test 2: Podcast with dictionary input
    print("Test 2: Podcast mode with dictionary input")
    input_data = {"sentences": podcast_sentences}
    context = {
        "podcast_mode": True,
        "min_pause_duration": 0.5,  # Longer pauses
        "max_pause_duration": 1.5,
        "speed": 1.1,  # Slightly faster
        "filename": "test_podcast_custom.wav"
    }
    
    result2 = await agent.process(input_data, context)
    print(f"Custom podcast audio file created: {result2.get('audio_file')}")
    print(f"Total text length: {result2.get('text_length')}")
    print()
    
    # Test 3: Two-person dialog podcast
    print("Test 3: Simple two-person dialog")
    dialog_sentences = [
        {"voice": "af_sarah", "text": "Welcome to our technology podcast! I'm Sarah."},
        {"voice": "am_michael", "text": "And I'm Michael. Today we're discussing artificial intelligence."},
        {"voice": "af_sarah", "text": "That's right! AI is revolutionizing so many industries."},
        {"voice": "am_michael", "text": "Absolutely. From healthcare to transportation, AI is everywhere."},
        {"voice": "af_sarah", "text": "What excites you most about AI development, Michael?"},
        {"voice": "am_michael", "text": "I think the potential for solving complex problems is incredible."},
        {"voice": "af_sarah", "text": "Great point! Thanks for joining us today, everyone."},
        {"voice": "am_michael", "text": "Until next time!"}
    ]
    
    context = {
        "podcast_mode": True,
        "min_pause_duration": 0.4,
        "max_pause_duration": 0.8,
        "speed": 0.95,  # Slightly slower for clear speech
        "process_id": "dialog_test"
    }
    
    result3 = await agent.process(dialog_sentences, context)
    print(f"Dialog podcast created: {result3.get('audio_file')}")
    print(f"Voice info: {json.dumps(result3.get('voice_info'), indent=2)}")
    print()
    
    # Test 4: Error handling - invalid input
    print("Test 4: Error handling - invalid input")
    try:
        context = {"podcast_mode": True}
        result4 = await agent.process("This is just a string, not sentences", context)
        print(f"Error result: {result4.get('error')}")
    except Exception as e:
        print(f"Expected error: {str(e)}")
    print()
    
    # Display available voices for reference
    print("Available voices for podcast creation:")
    for voice in KokoroTTSAgent.get_available_voices():
        print(f"  - {voice}")

async def demo_podcast_creation():
    """Demo function showing how to create a simple podcast"""
    
    print("=== Kokoro TTS Podcast Demo ===")
    
    # Simple configuration
    config = KokoroTTSAgent.get_default_config()
    agent = KokoroTTSAgent(config, "Podcast TTS Agent")
    
    # Create a simple news-style podcast
    news_podcast = [
        {"voice": "af_sarah", "text": "Good morning, and welcome to Tech News Daily. I'm Sarah."},
        {"voice": "am_michael", "text": "And I'm Michael. Let's dive into today's headlines."},
        {"voice": "af_sarah", "text": "Our top story: Advances in neural text-to-speech technology are making AI voices more natural than ever."},
        {"voice": "am_michael", "text": "That's fascinating, Sarah. These new models can capture nuances in human speech patterns."},
        {"voice": "af_sarah", "text": "Exactly! And the applications are endless - from audiobooks to virtual assistants."},
        {"voice": "am_michael", "text": "We'll continue following this story. That's all for today's Tech News Daily."},
        {"voice": "af_sarah", "text": "Thanks for listening, everyone!"}
    ]
    
    context = {
        "podcast_mode": True,
        "filename": "tech_news_demo.wav"
    }
    
    result = await agent.process(news_podcast, context)
    print(f"Demo podcast created: {result.get('audio_file')}")
    print(f"Duration: {len(news_podcast)} sentences")
    print(f"Voices: {', '.join(result.get('voices_used', []))}")

async def test_validation_errors():
    """Test input validation and error messages for podcast mode"""
    
    print("=== Testing Input Validation ===")
    
    # Configuration with podcast mode enabled
    config = {
        "voice": "af_sarah",
        "speed": 1.0,
        "language": "en-us",
        "podcast_mode": True,  # Enable podcast mode in config
        "model_path": "./model_files/kokkoro-82m/kokoro-v1.0.onnx",
        "voices_path": "./model_files/kokkoro-82m/voices/voices-v1.0.bin"
    }
    
    agent = KokoroTTSAgent(config, "Validation test agent")
    
    # Test 1: Empty input
    print("Test 1: Empty input")
    result = await agent.process([], {})
    print(f"Error: {result.get('error')}")
    if 'format_requirements' in result:
        print(f"Format requirements provided: {result['format_requirements']['input_format']}")
    print()
    
    # Test 2: Invalid sentence structure (missing voice)
    print("Test 2: Missing 'voice' field")
    invalid_sentences = [
        {"text": "Hello world!"}  # Missing voice field
    ]
    result = await agent.process(invalid_sentences, {})
    print(f"Error: {result.get('error')}")
    print()
    
    # Test 3: Invalid sentence structure (missing text)
    print("Test 3: Missing 'text' field")
    invalid_sentences = [
        {"voice": "af_sarah"}  # Missing text field
    ]
    result = await agent.process(invalid_sentences, {})
    print(f"Error: {result.get('error')}")
    print()
    
    # Test 4: Invalid voice name
    print("Test 4: Invalid voice name")
    invalid_sentences = [
        {"text": "Hello world!", "voice": "invalid_voice"}
    ]
    result = await agent.process(invalid_sentences, {})
    print(f"Error: {result.get('error')}")
    print()
    
    # Test 5: Wrong data type for sentence
    print("Test 5: Sentence is not a dictionary")
    invalid_sentences = [
        "This should be a dictionary"  # String instead of dict
    ]
    result = await agent.process(invalid_sentences, {})
    print(f"Error: {result.get('error')}")
    print()
    
    # Test 6: Empty text field
    print("Test 6: Empty text field")
    invalid_sentences = [
        {"text": "", "voice": "af_sarah"}  # Empty text
    ]
    result = await agent.process(invalid_sentences, {})
    print(f"Error: {result.get('error')}")
    print()
    
    # Test 7: Show format requirements
    print("Test 7: Format requirements example")
    result = await agent.process([], {})
    if 'format_requirements' in result:
        print("Available voices:", result['format_requirements']['available_voices'])
        print("Example format:")
        print(result['format_requirements']['example'])

if __name__ == "__main__":
    print("Testing Kokoro TTS Podcast Feature...")
    asyncio.run(test_podcast_mode())
    print("\n" + "="*50 + "\n")
    asyncio.run(test_validation_errors())
    print("\n" + "="*50 + "\n")
    asyncio.run(demo_podcast_creation())
    print("\n" + "="*50 + "\n")
    asyncio.run(test_validation_errors())
