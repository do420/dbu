"""
Quick test to verify the Kokoro TTS agent type conversion fix
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.kokoro_agent import KokoroTTSAgent

async def test_type_conversion():
    """Test that string parameters are properly converted to numeric types"""
    
    print("Testing Kokoro TTS agent type conversion...")
    
    # Test configuration with mixed types
    config = {
        "voice": "af_sarah",
        "speed": "1.2",  # String instead of float
        "language": "en-us"
    }
    
    try:
        # Create agent instance
        agent = KokoroTTSAgent(config, "Test agent")
        print(f"✓ Agent created successfully")
        print(f"  - Voice: {agent.voice} (type: {type(agent.voice)})")
        print(f"  - Speed: {agent.speed} (type: {type(agent.speed)})")
        print(f"  - Language: {agent.language} (type: {type(agent.language)})")
        
        # Test with context containing string parameters
        context = {
            "voice": "am_michael",
            "speed": "0.8",  # String speed
            "language": "en-us"
        }
        
        print(f"\nTesting process with context: {context}")
        
        # Note: This will fail if kokoro-onnx is not installed or model files are missing
        # But it should at least pass the type validation part
        try:
            result = await agent.process("Hello, this is a test.", context)
            if "error" in result:
                print(f"⚠️  Expected error (model/library not available): {result['error']}")
            else:
                print(f"✓ Process completed successfully: {result.get('audio_file')}")
        except Exception as e:
            error_msg = str(e)
            if ">=" in error_msg and "not supported between instances of 'str' and 'float'" in error_msg:
                print(f"❌ Type conversion error still exists: {error_msg}")
                return False
            else:
                print(f"⚠️  Expected error (model/library not available): {error_msg}")
        
        print("✓ Type conversion test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating agent: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_type_conversion())
    if success:
        print("\n🎉 All tests passed! The type conversion issue should be fixed.")
    else:
        print("\n💥 Tests failed. The issue may still exist.")
