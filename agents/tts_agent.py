from typing import Dict, Any, List, Optional
import os
import asyncio
import logging
import tempfile
from pathlib import Path
import edge_tts
from .base import BaseAgent

logger = logging.getLogger(__name__)

class TTSAgent(BaseAgent):
    """Agent that converts text to speech using Edge TTS"""
    
    # Class variable to store supported voices
    _supported_voices: List[Dict[str, str]] = []
    
    def __init__(self, config: Dict[str, Any], system_instruction: str):
        super().__init__(config, system_instruction)
        
        # Set default voice if not specified
        if "voice" not in self.config:
            self.config["voice"] = "en-US-ChristopherNeural"
        
        # Set default output directory
        self.output_dir = "_OUTPUT"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set rate/pitch if specified
        self.rate = self.config.get("rate", "+0%")
        self.volume = self.config.get("volume", "+0%")
        self.pitch = self.config.get("pitch", "+0Hz")
        
    @staticmethod
    async def list_voices() -> List[Dict[str, str]]:
        """Get a list of all available voices"""
        if not TTSAgent._supported_voices:
            try:
                raw_voices = await edge_tts.list_voices()
                # Convert the complex voice data to simple string format
                formatted_voices = []
                for voice in raw_voices:
                    # Convert VoiceTag from complex object to simple string
                    voice_tag = voice.get("VoiceTag", {})
                    if isinstance(voice_tag, dict):
                        # Extract meaningful info from VoiceTag and convert to string
                        content_categories = voice_tag.get("ContentCategories", [])
                        voice_personalities = voice_tag.get("VoicePersonalities", [])
                        tag_parts = []
                        if content_categories:
                            tag_parts.append(f"Categories: {', '.join(content_categories)}")
                        if voice_personalities:
                            tag_parts.append(f"Personalities: {', '.join(voice_personalities)}")
                        voice_tag_str = "; ".join(tag_parts) if tag_parts else "General"
                    else:
                        voice_tag_str = str(voice_tag) if voice_tag else "General"
                    
                    formatted_voice = {
                        "Name": str(voice.get("Name", "")),
                        "ShortName": str(voice.get("ShortName", "")),
                        "Gender": str(voice.get("Gender", "")),
                        "Locale": str(voice.get("Locale", "")),
                        "SuggestedCodec": str(voice.get("SuggestedCodec", "")),
                        "FriendlyName": str(voice.get("FriendlyName", "")),
                        "Status": str(voice.get("Status", "")),
                        "VoiceTag": voice_tag_str
                    }
                    formatted_voices.append(formatted_voice)
                
                TTSAgent._supported_voices = formatted_voices
            except Exception as e:
                logger.error(f"Failed to retrieve voice list: {str(e)}")
                return []
        return TTSAgent._supported_voices
    
    @staticmethod
    async def get_voice_by_language(language_code: str) -> Optional[str]:
        """Get a voice for a specific language"""
        voices = await TTSAgent.list_voices()
        filtered_voices = [v for v in voices if v["Locale"].startswith(language_code)]
        if filtered_voices:
            return filtered_voices[0]["ShortName"]
        return None
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert input text to speech and return path to audio file"""
        context = context or {}
        
        try:

                # Handle different input types
            if isinstance(input_data, dict):
                # If input is a dict (like RAG response), extract the text
                if "response" in input_data:
                    input_text = input_data["response"]
                elif "output" in input_data:
                    input_text = input_data["output"]
                elif "text" in input_data:
                    input_text = input_data["text"]
                else:
                    # If no recognizable text field, convert entire dict to string
                    input_text = str(input_data)
            elif isinstance(input_data, str):
                input_text = input_data
            else:
                # Convert any other type to string
                input_text = str(input_data)
            
            # Validate that we have actual text content
            if not input_text or not input_text.strip():
                raise ValueError("No valid text content found in input")
            
            # Get voice from config or context
            voice = context.get("voice", self.config["voice"])
            
            # If language code is provided instead of voice name, get an appropriate voice
            if len(voice) <= 5 and "-" in voice:  # Likely a language code like "en-US"
                detected_voice = await self.get_voice_by_language(voice)
                if detected_voice:
                    voice = detected_voice
                    logger.debug(f"Selected voice {voice} for language {voice}")
                else:
                    logger.warning(f"No voice found for language {voice}, using default")
            
            # Generate a unique filename with process_id if provided
            process_id = context.get("process_id")
            if process_id:
                filename = f"process_{process_id}.mp3"
            else:
                # Fallback to timestamp if no process_id
                filename = context.get("filename", f"tts_{asyncio.get_event_loop().time()}.mp3")
                if not filename.endswith(".mp3"):
                    filename += ".mp3"
            
            output_path = os.path.join(self.output_dir, filename)
            
            # Convert text to speech
            communicate = edge_tts.Communicate(input_text, voice, rate=self.rate, volume=self.volume, pitch=self.pitch)
            
            logger.debug(f"Converting text to speech with voice: {voice}")
            logger.debug(f"Output will be saved to: {output_path}")
            
            # Run the TTS conversion
            await communicate.save(output_path)
            
            # Get info about the voice used
            voices = await self.list_voices()
            voice_info = next((v for v in voices if v["ShortName"] == voice), {})
            
            # Construct the URL for the audio file
            audio_url = f"/audio/{filename}"
            
            return {
                "output": output_path,
                "audio_file": output_path,
                "audio_url": audio_url,
                "voice": voice,
                "voice_info": voice_info,
                "text_length": len(input_text),
                "agent_type": "tts",
                "token_usage": {}  
            }
        except Exception as e:
            logger.error(f"Error with TTS conversion: {str(e)}")
            return {
                "output": f"Error with TTS conversion: {str(e)}",
                "error": str(e),
                "agent_type": "tts",
                "token_usage": {}
            }
    
    @staticmethod
    async def get_supported_languages() -> List[str]:
        """Get a list of all supported language codes"""
        voices = await TTSAgent.list_voices()
        # Extract unique language codes
        languages = set(voice["Locale"] for voice in voices)
        return sorted(languages)