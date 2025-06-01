from typing import Dict, Any, List, Optional
import os
import asyncio
import logging
import numpy as np
import torch
from scipy.io.wavfile import write
from bark import generate_audio, preload_models
from .base import BaseAgent

logger = logging.getLogger(__name__)

class BarkTTSAgent(BaseAgent):
    """Agent that converts text to speech using Suno-Bark"""
    
    # Track whether models have been preloaded
    _models_preloaded = False
    
    def __init__(self, config: Dict[str, Any], system_instruction: str):
        super().__init__(config, system_instruction)
        
        # Set default output directory
        self.output_dir = self.config.get("output_dir", "_OUTPUT")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set voice parameters if specified
        self.voice_preset = self.config.get("voice_preset", None)
        self.text_temp = self.config.get("text_temp", 0.7)
        self.waveform_temp = self.config.get("waveform_temp", 0.7)
        
        # Sample rate for Bark
        self.sample_rate = 24000
        
        # Preload models if not already done
        if not BarkTTSAgent._models_preloaded:
            logger.info("Preloading Bark models for first use...")
            # This line can be commented out for faster loading if you have limited GPU memory
            # preload_models()
            BarkTTSAgent._models_preloaded = True
        
    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert input text to speech using Bark and return path to audio file"""
        context = context or {}
        
        try:
            # Generate a unique filename with process_id if provided
            process_id = context.get("process_id")
            if process_id:
                filename = f"process_{process_id}.wav"
            else:
                # Fallback to timestamp if no process_id
                filename = context.get("filename", f"bark_{asyncio.get_event_loop().time()}.wav")
                if not filename.endswith(".wav"):
                    filename += ".wav"
            
            output_path = os.path.join(self.output_dir, filename)
            
            logger.debug(f"Converting text to speech with Bark")
            logger.debug(f"Output will be saved to: {output_path}")
            
            # Handle voice preset
            voice_preset = context.get("voice_preset", self.voice_preset)
            
            # Run the TTS conversion in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            audio_array = await loop.run_in_executor(
                None,
                lambda: self._generate_audio(input_text, voice_preset)
            )
            
            # Save the audio file
            await loop.run_in_executor(
                None,
                lambda: write(output_path, self.sample_rate, (audio_array * 32767).astype(np.int16))
            )
            
            # Construct the URL for the audio file
            audio_url = f"/audio/{filename}"
            
            return {
                "output": output_path,
                "audio_file": output_path,
                "audio_url": audio_url,
                "voice_preset": voice_preset,
                "text_length": len(input_text),
                "agent_type": "bark_tts",
                "token_usage": {}  # Approximate token count
            }
        except Exception as e:
            logger.error(f"Error with Bark TTS conversion: {str(e)}")
            return {
                "output": f"Error with Bark TTS conversion: {str(e)}",
                "error": str(e),
                "agent_type": "bark_tts",
                "token_usage": {"total_tokens": 0}
            }
    
    def _generate_audio(self, text: str, voice_preset: Optional[str] = None) -> np.ndarray:
        """Generate audio using Bark, splitting by sentences and adding silence between them"""
        try:
            import re
            import tempfile
            import shutil
            import time
            import random
            
            # Create a temporary directory for sentence audio files
            temp_dir = tempfile.mkdtemp(prefix="bark_temp_")
            
            try:
                # Set environment variable for voice preset if specified
                old_env = None
                if voice_preset:
                    old_env = os.environ.get("BARK_VOICE", None)
                    os.environ["BARK_VOICE"] = voice_preset
                    logger.debug(f"Set BARK_VOICE to {voice_preset}")
                
                # Split text into sentences
                # This regex matches sentence endings followed by space or end of string
                sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if not sentences:
                    sentences = [text]  # Fallback if no sentences were detected
                
                logger.debug(f"Split text into {len(sentences)} sentences")
                
                # Process each sentence separately
                audio_segments = []
                
                for i, sentence in enumerate(sentences):
                    logger.debug(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:30]}...")
                    
                    # Generate audio for this sentence
                    sentence_audio = generate_audio(sentence, history_prompt="v2/en_speaker_6", text_temp = 0.5)
                    audio_segments.append(sentence_audio)
                    
                    # Add random silence between sentences (except after the last one)
                    if i < len(sentences) - 1:
                        # Generate random silence duration between 0.3 and 0.7 seconds
                        silence_duration = random.uniform(0.3, 0.7)
                        silence_samples = int(silence_duration * self.sample_rate)
                        silence = np.zeros(silence_samples)
                        audio_segments.append(silence)
                
                # Concatenate all audio segments
                full_audio = np.concatenate(audio_segments)
                
                # Restore old environment variable
                if voice_preset and old_env is not None:
                    os.environ["BARK_VOICE"] = old_env
                elif voice_preset:
                    os.environ.pop("BARK_VOICE", None)
                
                return full_audio
                
            finally:
                # Clean up the temporary directory
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary directory: {cleanup_error}")
                    
        except Exception as e:
            logger.error(f"Error in Bark audio generation: {str(e)}")
            # Try basic generation as fallback
            logger.info("Falling back to basic audio generation")
            return generate_audio(text)
    
    @staticmethod
    def get_available_voices() -> List[str]:
        """Get a list of available voice presets in Bark"""
        # Bark has predefined voice presets
        # This is a simplified list - in a production system you might need to update this
        return [
            "v2/en_speaker_0",
            "v2/en_speaker_1", 
            "v2/en_speaker_2",
            "v2/en_speaker_3",
            "v2/en_speaker_4",
            "v2/en_speaker_5",
            "v2/en_speaker_6",
            "v2/en_speaker_7",
            "v2/en_speaker_8",
            "v2/en_speaker_9",
            # Additional languages
            "v2/de_speaker_0",
            "v2/es_speaker_0",
            "v2/fr_speaker_0",
            "v2/it_speaker_0",
            "v2/ja_speaker_0",
            "v2/ko_speaker_0",
            "v2/pl_speaker_0",
            "v2/pt_speaker_0",
            "v2/ru_speaker_0",
            "v2/tr_speaker_0",
            "v2/zh_speaker_0"
        ]