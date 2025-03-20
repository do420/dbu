from typing import Dict, Any, List, Optional
import os
import asyncio
import logging
import tempfile
from pathlib import Path
import whisperx
from .base import BaseAgent

logger = logging.getLogger(__name__)

class TranscribeAgent(BaseAgent):
    """Agent that transcribes audio/video files using WhisperX"""
    
    # Track loaded models to avoid reloading
    _models = {}
    
    def __init__(self, config: Dict[str, Any], system_instruction: str):
        super().__init__(config, system_instruction)
        
        # Set default output directory
        self.output_dir = self.config.get("output_dir", "_OUTPUT")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set transcription parameters
        self.model_name = self.config.get("model_name", "large-v3")
        self.device = self.config.get("device", "cuda")  # 'cuda' or 'cpu'
        self.compute_type = self.config.get("compute_type", "float16")  # 'float16', 'float32', 'int8'
        
        # Whether to align text or not
        self.align_text = self.config.get("align_text", False)
        self.align_model = self.config.get("align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H")
        
        # Whether to include timestamps in output
        self.include_timestamps = self.config.get("include_timestamps", False)
        
    async def process(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transcribe audio file and return the transcription
        
        Args:
            input_data: Dict containing:
                - file_path: Path to audio file
                - language: Language code (optional, defaults to 'en')
            context: Additional context
                
        Returns:
            Dict with transcription and details
        """
        context = context or {}
        
        # Get audio file path and language
        if isinstance(input_data, dict):
            file_path = input_data.get("file_name")
            language = input_data.get("language", "en")
        else:
            # If input is a string, assume it's the file path
            file_path = input_data
            language = context.get("language", "en")
        
        #file path is not output + file_path
        file_path = os.path.join("_INPUT", file_path)


        if not file_path or not os.path.exists(file_path):
            return {
                "error": f"Input audio file not found: {file_path}",
                "output": f"Error: Input audio file not found"
            }
        
        # Generate output filenames based on process_id if available
        process_id = context.get("process_id")
        if process_id:
            basename = f"transcription_process_{process_id}"
        else:
            # Fallback to audio filename if no process_id
            basename = f"transcription_{os.path.basename(file_path).rsplit('.', 1)[0]}"
        
        txt_output_path = os.path.join(self.output_dir, f"{basename}.txt")
        json_output_path = os.path.join(self.output_dir, f"{basename}.json")
        
        try:
            # Run the transcription in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_audio,
                file_path, language, txt_output_path, json_output_path
            )
            
            # Add file URL to result
            txt_url = f"/_OUTPUT/{os.path.basename(txt_output_path)}"
            json_url = f"/_OUTPUT/{os.path.basename(json_output_path)}"
            
            result["txt_url"] = txt_url
            result["json_url"] = json_url
            result["output"] = result["transcription"]  # Set output field for compatibility
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {
                "error": str(e),
                "output": f"Error transcribing audio: {str(e)}",
                "agent_type": "transcribe"
            }
    
    def _transcribe_audio(self, file_path: str, language: str, 
                         txt_output_path: str, json_output_path: str) -> Dict[str, Any]:
        """Synchronous implementation of audio transcription"""
        import json
        
        # Load or get the model
        model_key = f"{self.model_name}_{self.device}_{self.compute_type}"
        if model_key not in self._models:
            logger.info(f"Loading WhisperX model {self.model_name} on {self.device}...")
            model = whisperx.load_model(
                self.model_name, 
                self.device, 
                compute_type=self.compute_type
            )
            self._models[model_key] = model
        else:
            logger.debug(f"Using cached WhisperX model {model_key}")
            model = self._models[model_key]
        
        # Perform transcription
        logger.info(f"Transcribing {file_path} in language {language}")
        result = model.transcribe(file_path, language=language)
        
        # Format segments for output
        segments = result["segments"]
        
        # Extract the transcription text
        if self.include_timestamps:
            # Include timestamps in the output
            lines = []
            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                
                start_time_str = self._format_time(start_time)
                end_time_str = self._format_time(end_time)
                
                lines.append(f"[{start_time_str} --> {end_time_str}] {text}")
            transcription_text = "\n".join(lines)
        else:
            # Just extract the text without timestamps
            transcription_text = "\n".join(segment["text"] for segment in segments)
        
        # Perform alignment if requested
        if self.align_text:
            logger.info(f"Aligning transcription with audio...")
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=language,
                    device=self.device,
                    model_name=self.align_model
                )
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    file_path,
                    self.device,
                    return_char_alignments=True
                )
                logger.info(f"Alignment complete")
            except Exception as align_error:
                logger.warning(f"Alignment failed: {str(align_error)}")
        
        # Save the transcription to the text file
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write(transcription_text)
        
        # Save the full result to the JSON file
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Transcription saved to {txt_output_path}")
        
        return {
            "transcription": transcription_text,
            "segments": segments,
            "language": language,
            "txt_path": txt_output_path,
            "json_path": json_output_path,
            "duration": result.get("duration", 0),
            "agent_type": "transcribe"
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into HH:MM:SS.ms format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    @staticmethod
    def supported_models() -> List[str]:
        """Get a list of supported WhisperX models"""
        return [
            "large-v3",
            "large-v2", 
            "medium",
            "small",
            "base",
            "tiny"
        ]
    
    @staticmethod
    def supported_languages() -> Dict[str, str]:
        """Get a mapping of language codes to names"""
        return {
            "en": "English",
            "zh": "Chinese",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "ko": "Korean",
            "fr": "French",
            "ja": "Japanese",
            "pt": "Portuguese",
            "tr": "Turkish",
            "pl": "Polish",
            "ca": "Catalan",
            "nl": "Dutch",
            "ar": "Arabic",
            "sv": "Swedish",
            "it": "Italian",
            "id": "Indonesian",
            "hi": "Hindi",
            "fi": "Finnish",
            "vi": "Vietnamese",
            "he": "Hebrew",
            "uk": "Ukrainian",
            "el": "Greek",
            "ms": "Malay",
            "cs": "Czech",
            "ro": "Romanian",
            "da": "Danish",
            "hu": "Hungarian",
            "ta": "Tamil",
            "no": "Norwegian",
            "th": "Thai",
            "ur": "Urdu",
            "hr": "Croatian",
            "bg": "Bulgarian",
            "lt": "Lithuanian",
            "la": "Latin",
            "mi": "Maori",
            "ml": "Malayalam",
            "cy": "Welsh",
            "sk": "Slovak",
            "te": "Telugu",
            "fa": "Persian",
            "lv": "Latvian",
            "bn": "Bengali",
            "sr": "Serbian",
            "az": "Azerbaijani",
            "sl": "Slovenian",
            "kn": "Kannada",
            "et": "Estonian",
            "mk": "Macedonian",
            "br": "Breton",
            "eu": "Basque",
            "is": "Icelandic",
            "hy": "Armenian",
            "ne": "Nepali",
            "mn": "Mongolian",
            "bs": "Bosnian",
            "kk": "Kazakh",
            "sq": "Albanian",
            "sw": "Swahili",
            "gl": "Galician",
            "mr": "Marathi",
            "pa": "Punjabi",
            "si": "Sinhala",
            "km": "Khmer",
            "sn": "Shona",
            "yo": "Yoruba",
            "so": "Somali",
            "af": "Afrikaans",
            "oc": "Occitan",
            "ka": "Georgian",
            "be": "Belarusian",
            "tg": "Tajik",
            "sd": "Sindhi",
            "gu": "Gujarati",
            "am": "Amharic",
            "yi": "Yiddish",
            "lo": "Lao",
            "uz": "Uzbek",
            "fo": "Faroese",
            "ht": "Haitian Creole",
            "ps": "Pashto",
            "tk": "Turkmen",
            "nn": "Nynorsk",
            "mt": "Maltese",
            "sa": "Sanskrit",
            "lb": "Luxembourgish",
            "my": "Myanmar",
            "bo": "Tibetan",
            "tl": "Tagalog",
            "mg": "Malagasy",
            "as": "Assamese",
            "tt": "Tatar",
            "haw": "Hawaiian",
            "ln": "Lingala",
            "ha": "Hausa",
            "ba": "Bashkir",
            "jw": "Javanese",
            "su": "Sundanese",
        }