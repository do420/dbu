from typing import Dict, Any, List, Optional
import os
import asyncio
import logging
import numpy as np
import soundfile as sf
from .base import BaseAgent

logger = logging.getLogger(__name__)

class KokoroTTSAgent(BaseAgent):
    """Agent that converts text to speech using Kokoro ONNX"""
    
    # Track whether models have been loaded
    _kokoro_instance = None
    def __init__(self, config: Dict[str, Any], system_instruction: str):
        super().__init__(config, system_instruction)
        
        # Set default output directory
        self.output_dir = self.config.get("output_dir", "_OUTPUT")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set voice parameters
        self.voice = self.config.get("voice", "af_sarah") # ['af_sarah', 'af_nicole', 'af_sky', 'am_adam', 'am_michael', 'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis']
        
        # Ensure speed is a float (convert from string if necessary)
        speed_config = self.config.get("speed", 1.0)
        if isinstance(speed_config, str):
            try:
                self.speed = float(speed_config)
            except ValueError:
                logger.warning(f"Invalid speed value in config '{speed_config}', using default 1.0")
                self.speed = 1.0
        elif isinstance(speed_config, (int, float)):
            self.speed = float(speed_config)
        else:
            logger.warning(f"Invalid speed type in config {type(speed_config)}, using default 1.0")
            self.speed = 1.0
            
        self.language = self.config.get("language", "en-us") # ['en-us', 'en-gb', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh']
        
        # Podcast mode configuration (moved from context)
        self.podcast_mode = self.config.get("podcast_mode", False)
        
        # Model file paths
        self.model_path = self.config.get("model_path", "./model_files/kokkoro-82m/kokoro-v1.0.onnx")
        self.voices_path = self.config.get("voices_path", "./model_files/kokkoro-82m/voices/voices-v1.0.bin")
        
        # Sample rate for Kokoro (will be set when model is loaded)
        self.sample_rate = 24000  # Default, will be updated from actual model
        
        # Initialize Kokoro if not already done
        if KokoroTTSAgent._kokoro_instance is None:
            self._initialize_kokoro()
    
    def _initialize_kokoro(self):
        """Initialize the Kokoro TTS model"""
        try:
            from kokoro_onnx import Kokoro
            
            logger.info("Initializing Kokoro TTS model...")
            KokoroTTSAgent._kokoro_instance = Kokoro(self.model_path, self.voices_path)
            logger.info("Kokoro TTS model initialized successfully")
            
        except ImportError:
            logger.error("kokoro-onnx not installed. Please install with: pip install -U kokoro-onnx soundfile")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS model: {str(e)}")
            raise
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert input text to speech using Kokoro and return path to audio file"""
        context = context or {}
        
        try:
            # Check if this is podcast mode (from config, not context)
            is_podcast_mode = self.podcast_mode
            
            if is_podcast_mode:
                return await self._process_podcast(input_data, context)
            else:
                return await self._process_single_text(input_data, context)
                
        except ValueError as e:
            # Return detailed format requirements for validation errors
            if "podcast mode requires" in str(e).lower() or "sentence" in str(e).lower():
                return {
                    "output": f"Input format error: {str(e)}",
                    "error": str(e),
                    "format_requirements": self._get_podcast_format_requirements(),
                    "agent_type": "kokoro_tts",
                    "token_usage": {}
                }
            else:
                return {
                    "output": f"Input validation error: {str(e)}",
                    "error": str(e),
                    "agent_type": "kokoro_tts",
                    "token_usage": {}
                }
        except Exception as e:
            logger.error(f"Error with Kokoro TTS conversion: {str(e)}")
            return {
                "output": f"Error with Kokoro TTS conversion: {str(e)}",
                "error": str(e),
                "agent_type": "kokoro_tts",
                "token_usage": {}
            }
    
    async def _process_single_text(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process single text input (original functionality)"""
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
        voice = context.get("voice", self.voice)
        speed = context.get("speed", self.speed)
        language = context.get("language", self.language)
        
        # Ensure speed is a float (convert from string if necessary)
        if isinstance(speed, str):
            try:
                speed = float(speed)
            except ValueError:
                logger.warning(f"Invalid speed value '{speed}', using default {self.speed}")
                speed = self.speed
        elif not isinstance(speed, (int, float)):
            logger.warning(f"Invalid speed type {type(speed)}, using default {self.speed}")
            speed = self.speed
        
        # Generate a unique filename with process_id if provided
        process_id = context.get("process_id")
        if process_id:
            filename = f"process_{process_id}.wav"
        else:
            # Fallback to timestamp if no process_id
            filename = context.get("filename", f"kokoro_{asyncio.get_event_loop().time()}.wav")
            if not filename.endswith(".wav"):
                filename += ".wav"
        
        output_path = os.path.join(self.output_dir, filename)
        
        logger.debug(f"Converting text to speech with Kokoro")
        logger.debug(f"Voice: {voice}, Speed: {speed}, Language: {language}")
        logger.debug(f"Output will be saved to: {output_path}")
        
        # Run the TTS conversion in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        audio_data, sample_rate = await loop.run_in_executor(
            None,
            self._generate_audio,
            input_text, voice, speed, language
        )
        
        # Save the audio file
        await loop.run_in_executor(
            None,
            lambda: sf.write(output_path, audio_data, sample_rate)
        )
        
        # Update the actual sample rate
        self.sample_rate = sample_rate
        
        # Construct the URL for the audio file
        audio_url = f"/audio/{filename}"
        
        return {
            "output": output_path,
            "audio_file": output_path,
            "audio_url": audio_url,
            "voice": voice,
            "voice_info": {"voice": voice, "speed": speed, "language": language},
            "text_length": len(input_text),
            "agent_type": "kokoro_tts",
            "sample_rate": sample_rate,
            "token_usage": {}
        }
    async def _process_podcast(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process podcast mode with multiple sentences and voices"""
        import random
        
        # Detailed input validation for podcast mode
        try:
            sentences = self._validate_podcast_input(input_data)
        except ValueError as e:
            # Re-raise with detailed format information
            raise ValueError(f"{str(e)}\n\nRequired format:\n{self._get_podcast_format_example()}")
        
        # Get podcast settings
        min_pause_duration = context.get("min_pause_duration", 0.3)
        max_pause_duration = context.get("max_pause_duration", 1.0)
        speed = context.get("speed", self.speed)
        
        # Ensure speed is a float
        if isinstance(speed, str):
            try:
                speed = float(speed)
            except ValueError:
                logger.warning(f"Invalid speed value '{speed}', using default {self.speed}")
                speed = self.speed
        elif not isinstance(speed, (int, float)):
            logger.warning(f"Invalid speed type {type(speed)}, using default {self.speed}")
            speed = self.speed
        
        # Generate filename
        process_id = context.get("process_id")
        if process_id:
            filename = f"process_{process_id}.wav"
        else:
            filename = context.get("filename", f"process_{asyncio.get_event_loop().time()}.wav")
            if not filename.endswith(".wav"):
                filename += ".wav"
        
        output_path = os.path.join(self.output_dir, filename)
        
        logger.info(f"Creating podcast with {len(sentences)} sentences")
        logger.debug(f"Podcast will be saved to: {output_path}")
        
        # Generate podcast audio
        loop = asyncio.get_event_loop()
        audio_data, sample_rate = await loop.run_in_executor(
            None,
            self._generate_podcast_audio,
            sentences, speed, min_pause_duration, max_pause_duration
        )
        
        # Save the audio file
        await loop.run_in_executor(
            None,
            lambda: sf.write(output_path, audio_data, sample_rate)
        )
        
        # Calculate total text length
        total_text_length = sum(len(sentence["text"]) for sentence in sentences)
        
        # Get unique voices used
        voices_used = list(set(sentence["voice"] for sentence in sentences))
        
        # Construct the URL for the audio file
        audio_url = f"/audio/{filename}"
        
        return {
            "output": output_path,
            "audio_file": output_path,
            "audio_url": audio_url,
            "voices_used": voices_used,
            "voice_info": {
                "mode": "podcast",
                "voices_used": voices_used,
                "sentence_count": len(sentences),
                "speed": speed,
                "language": "en-us"  # Always English for podcast mode
            },
            "text_length": total_text_length,
            "sentence_count": len(sentences),
            "agent_type": "kokoro_tts",
            "sample_rate": sample_rate,
            "token_usage": {}
        }
    
    def _generate_audio(self, text: str, voice: str, speed: float, language: str) -> tuple:
        """Generate audio using Kokoro TTS"""
        try:
            if KokoroTTSAgent._kokoro_instance is None:
                self._initialize_kokoro()
            
            logger.debug(f"Generating audio for text: {text[:50]}...")
            logger.debug(f"Parameters - voice: {voice} (type: {type(voice)}), speed: {speed} (type: {type(speed)}), language: {language} (type: {type(language)})")
            
            # Ensure all parameters are the correct types
            if not isinstance(speed, (int, float)):
                logger.error(f"Speed must be numeric, got {type(speed)}: {speed}")
                raise ValueError(f"Speed must be numeric, got {type(speed)}: {speed}")
            
            # Generate audio using Kokoro
            samples, sample_rate = KokoroTTSAgent._kokoro_instance.create(
                text,
                voice=voice,
                speed=float(speed),  # Ensure speed is float
                lang=language,
            )
            
            logger.debug(f"Generated audio with shape: {samples.shape}, sample rate: {sample_rate}")
            
            return samples, sample_rate
            
        except Exception as e:
            logger.error(f"Error in Kokoro audio generation: {str(e)}")
            raise
    
    def _generate_podcast_audio(self, sentences: List[Dict[str, str]], speed: float, 
                               min_pause_duration: float, max_pause_duration: float) -> tuple:
        """Generate podcast audio with multiple voices and automatic pauses"""
        import random
        
        try:
            if KokoroTTSAgent._kokoro_instance is None:
                self._initialize_kokoro()
            
            logger.info(f"Generating podcast audio with {len(sentences)} sentences")
            
            audio_segments = []
            sample_rate = None
            
            for i, sentence in enumerate(sentences):
                voice = sentence["voice"]
                text = sentence["text"]
                
                logger.debug(f"Processing sentence {i+1}/{len(sentences)} with voice {voice}: {text[:50]}...")
                
                # Generate audio for this sentence
                samples, current_sample_rate = KokoroTTSAgent._kokoro_instance.create(
                    text,
                    voice=voice,
                    speed=float(speed),
                    lang="en-us",  # Always English for podcast mode
                )
                
                # Set sample rate from first sentence
                if sample_rate is None:
                    sample_rate = current_sample_rate
                
                audio_segments.append(samples)
                
                # Add random pause between sentences (except after the last one)
                if i < len(sentences) - 1:
                    pause_audio = self._random_pause(min_pause_duration, max_pause_duration, sample_rate)
                    audio_segments.append(pause_audio)
            
            # Concatenate all audio segments
            full_audio = np.concatenate(audio_segments)
            
            logger.info(f"Generated podcast audio with total duration: {len(full_audio) / sample_rate:.2f} seconds")
            
            return full_audio, sample_rate
            
        except Exception as e:
            logger.error(f"Error in podcast audio generation: {str(e)}")
            raise
    def _random_pause(self, min_duration: float, max_duration: float, sample_rate: int) -> np.ndarray:
        """Generate random silence between sentences"""
        import random
        
        silence_duration = random.uniform(min_duration, max_duration)
        silence_samples = int(silence_duration * sample_rate)
        silence = np.zeros(silence_samples)
        logger.debug(f"Generated {silence_duration:.2f}s pause ({silence_samples} samples)")
        return silence
    
    def _validate_podcast_input(self, input_data: Any) -> List[Dict[str, str]]:
        """Validate input data for podcast mode and return sentences list"""
        import json
        
        sentences = []
        
        # Handle string input that might be JSON or markdown code block
        if isinstance(input_data, str):
            logger.debug(f"Processing string input. Length: {len(input_data)}")
            logger.debug(f"Input starts with: {repr(input_data[:50])}")
            logger.debug(f"Input ends with: {repr(input_data[-50:])}")
            
            # More flexible markdown code block detection
            # Handle various patterns: ```json\n...``` or ```json\n...\n``` or ```json\n...\n```\n
            if "```json" in input_data and "```" in input_data[input_data.find("```json") + 7:]:
                logger.debug("Detected markdown code block with ```json")
                
                # Find the start and end of the JSON content
                start_marker = input_data.find("```json")
                if start_marker != -1:
                    # Find where the JSON content actually starts (after ```json and possible newline)
                    json_start = start_marker + 7  # Length of "```json"
                    if json_start < len(input_data) and input_data[json_start] == '\n':
                        json_start += 1  # Skip the newline after ```json
                    
                    # Find the closing ```
                    end_marker = input_data.find("```", json_start)
                    if end_marker != -1:
                        json_content = input_data[json_start:end_marker]
                        logger.debug(f"Extracted JSON content: {repr(json_content[:100])}...")
                        
                        try:
                            parsed_data = json.loads(json_content)
                            logger.debug(f"Successfully parsed JSON. Type: {type(parsed_data)}")
                            
                            if isinstance(parsed_data, dict) and "sentences" in parsed_data:
                                sentences = parsed_data["sentences"]
                                logger.debug(f"Found {len(sentences)} sentences in 'sentences' field")
                            elif isinstance(parsed_data, list):
                                sentences = parsed_data
                                logger.debug(f"Using direct list with {len(sentences)} items")
                            else:
                                raise ValueError("Parsed JSON must contain 'sentences' field or be a list of sentence objects")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {str(e)}")
                            logger.error(f"Problematic JSON content: {repr(json_content)}")
                            raise ValueError(f"Invalid JSON format in markdown code block: {str(e)}")
                    else:
                        logger.error("Could not find closing ``` marker")
                        raise ValueError("Malformed markdown code block: missing closing ```")
                else:
                    logger.error("Could not find ```json marker")
                    raise ValueError("Malformed markdown code block: missing ```json")
            
            # Check if it's a simple markdown code block with "json\n" prefix
            elif input_data.startswith("json\n"):
                logger.debug("Detected simple markdown code block with json\\n prefix")
                # Remove the "json\n" prefix and parse the JSON
                json_content = input_data[5:]  # Remove "json\n"
                logger.debug(f"JSON content after removing prefix: {repr(json_content[:100])}...")
                
                try:
                    parsed_data = json.loads(json_content)
                    logger.debug(f"Successfully parsed JSON. Type: {type(parsed_data)}")
                    
                    if isinstance(parsed_data, dict) and "sentences" in parsed_data:
                        sentences = parsed_data["sentences"]
                        logger.debug(f"Found {len(sentences)} sentences in 'sentences' field")
                    elif isinstance(parsed_data, list):
                        sentences = parsed_data
                        logger.debug(f"Using direct list with {len(sentences)} items")
                    else:
                        raise ValueError("Parsed JSON must contain 'sentences' field or be a list of sentence objects")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                    logger.error(f"Problematic JSON content: {repr(json_content)}")
                    raise ValueError(f"Invalid JSON format in simple code block: {str(e)}")
            
            else:
                logger.debug("Attempting to parse as direct JSON string")
                # Try to parse as direct JSON string
                try:
                    parsed_data = json.loads(input_data)
                    logger.debug(f"Successfully parsed as direct JSON. Type: {type(parsed_data)}")
                    
                    if isinstance(parsed_data, dict) and "sentences" in parsed_data:
                        sentences = parsed_data["sentences"]
                        logger.debug(f"Found {len(sentences)} sentences in 'sentences' field")
                    elif isinstance(parsed_data, list):
                        sentences = parsed_data
                        logger.debug(f"Using direct list with {len(sentences)} items")
                    else:
                        raise ValueError("Parsed JSON must contain 'sentences' field or be a list of sentence objects")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse as direct JSON: {str(e)}")
                    logger.error(f"Input string: {repr(input_data[:200])}...")
                    raise ValueError("String input must be valid JSON, markdown code block starting with '```json', or simple code block starting with 'json\\n'")
        
        # Extract sentences from input data (existing logic for dict/list)
        elif isinstance(input_data, dict):
            logger.debug("Processing dictionary input")
            if "sentences" in input_data:
                sentences = input_data["sentences"]
                logger.debug(f"Found {len(sentences)} sentences in dict")
            else:
                logger.error("Dictionary input missing 'sentences' field")
                raise ValueError("Podcast mode requires input to be a dict with 'sentences' field or a list of sentence objects")
        elif isinstance(input_data, list):
            logger.debug(f"Processing list input with {len(input_data)} items")
            sentences = input_data
        else:
            logger.error(f"Unsupported input type: {type(input_data)}")
            raise ValueError("Podcast mode requires input to be either a dict with 'sentences' field, a list of sentence objects, or a JSON string")
        
        logger.debug(f"Final sentences count: {len(sentences)}")
        
        # Validate sentences structure
        if not sentences:
            raise ValueError("Sentences list cannot be empty")
        
        if not isinstance(sentences, list):
            raise ValueError("Sentences must be a list")
        
        available_voices = self.get_available_voices()
        
        for i, sentence in enumerate(sentences):
            logger.debug(f"Validating sentence {i+1}: {sentence}")
            
            # Check if sentence is a dictionary
            if not isinstance(sentence, dict):
                raise ValueError(f"Sentence {i+1} must be a dictionary with 'text' and 'voice' fields")
            
            # Check required fields
            if "text" not in sentence:
                raise ValueError(f"Sentence {i+1} is missing required 'text' field")
            
            if "voice" not in sentence:
                raise ValueError(f"Sentence {i+1} is missing required 'voice' field")
            
            # Validate field types and content
            if not isinstance(sentence["text"], str):
                raise ValueError(f"Sentence {i+1}: 'text' field must be a string")
            
            if not isinstance(sentence["voice"], str):
                raise ValueError(f"Sentence {i+1}: 'voice' field must be a string")
            
            if not sentence["text"].strip():
                raise ValueError(f"Sentence {i+1}: 'text' field cannot be empty")
            
            if sentence["voice"] not in available_voices:
                raise ValueError(f"Sentence {i+1}: voice '{sentence['voice']}' is not available. Available voices: {', '.join(available_voices)}")
        
        logger.info(f"Successfully validated {len(sentences)} sentences for podcast mode")
        return sentences
    def _get_podcast_format_requirements(self) -> Dict[str, Any]:
        """Get detailed format requirements for podcast mode"""
        return {
            "input_format": "JSON object with 'sentences' field, list of sentence objects, JSON string, or markdown code block starting with 'json\\n' or '```json\\n'",
            "sentence_structure": {
                "text": "string (required) - The text to be spoken",
                "voice": "string (required) - Voice to use for this sentence"
            },
            "available_voices": self.get_available_voices(),
            "example": self._get_podcast_format_example()
        }
    def _get_podcast_format_example(self) -> str:
        """Get a formatted example of correct podcast input"""
        return '''Format 1 (JSON object with sentences field):
{
  "sentences": [
    {"text": "Hello, welcome to our podcast!", "voice": "af_sarah"},
    {"text": "Today we'll discuss artificial intelligence.", "voice": "am_adam"},
    {"text": "It's a fascinating topic indeed.", "voice": "af_sarah"}
  ]
}

Format 2 (Direct list of sentences):
[
  {"text": "Hello, welcome to our podcast!", "voice": "af_sarah"},
  {"text": "Today we'll discuss artificial intelligence.", "voice": "am_adam"},
  {"text": "It's a fascinating topic indeed.", "voice": "af_sarah"}
]

Format 3 (Full markdown code block):
```json
{
  "sentences": [
    {"text": "Hello, welcome to our podcast!", "voice": "af_sarah"},
    {"text": "Today we'll discuss artificial intelligence.", "voice": "am_adam"},
    {"text": "It's a fascinating topic indeed.", "voice": "af_sarah"}
  ]
}
```

Format 4 (Simple markdown code block):
json
{
  "sentences": [
    {"text": "Hello, welcome to our podcast!", "voice": "af_sarah"},
    {"text": "Today we'll discuss artificial intelligence.", "voice": "am_adam"},
    {"text": "It's a fascinating topic indeed.", "voice": "af_sarah"}
  ]
}

Format 5 (JSON string):
"{\\"sentences\\":[{\\"text\\":\\"Hello!\\",\\"voice\\":\\"af_sarah\\"}]}"

Available voices: af_sarah, af_nicole, af_sky, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis'''

    @staticmethod
    def get_available_voices() -> List[str]:
        """Get a list of available voice presets in Kokoro"""
        # Based on the testing file, these are common Kokoro voices
        return [
            "af_sarah",
            "af_nicole", 
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis"
        ]
    
    @staticmethod
    def get_supported_languages() -> List[str]:
        """Get a list of supported language codes"""
        return [
            "en-us",
            "en-gb", 
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ja",
            "ko",
            "zh"
        ]
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for Kokoro TTS"""
        return {
            "voice": "af_sarah",
            "speed": 1.0,
            "language": "en-us",
            "podcast_mode": False,
            "model_path": "./model_files/kokkoro-82m/kokoro-v1.0.onnx",
            "voices_path": "./model_files/kokkoro-82m/voices/voices-v1.0.bin"
        }
