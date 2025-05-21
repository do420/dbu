from typing import Dict, Any
import logging
from deep_translator import GoogleTranslator
from .base import BaseAgent

logger = logging.getLogger(__name__)

class GoogleTranslateAgent(BaseAgent):
    """Agent that translates text using Google Translate via deep-translator library"""
    
    def __init__(self, config: Dict[str, Any], system_instruction: str):
        super().__init__(config, system_instruction)
        # Set default target language if not specified
        self.default_target_lang = self.config.get("target_language", "en")
    
    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Translate input text to the target language using Google Translate
        
        Args:
            input_text: Text to be translated
            context: Optional context data that may include:
                - target_language: Target language code (e.g., 'en', 'tr', 'es')
                
        Returns:
            Dict with the translated text
        """
        context = context or {}
        
        if not input_text or input_text.strip() == "":
            return {
                "output": "",
                "error": "Empty input text",
                "agent_type": "google_translate",
                "source_language": None,
                "target_language": None
            }
        
        try:
            # Get target language from context or use default
            target_language = context.get("target_language", self.default_target_lang)
            
            logger.debug(f"Translating text to {target_language}")
            
            # Initialize translator with auto-detection for source language
            translator = GoogleTranslator(source='auto', target=target_language)
            
            # Translate the text
            translated_text = translator.translate(input_text)
            
            # Try to detect the source language
            detected_source = translator.source
            if detected_source == 'auto':
                # If source is still 'auto', make a direct attempt to detect
                try:
                    from deep_translator import single_detection
                    detected_source = single_detection(input_text, api_key=self.config.get("api_key"))
                except:
                    detected_source = "unknown"
            
            logger.debug(f"Translation complete. Source language: {detected_source}")
            
            return {
                "output": translated_text,
                "agent_type": "google_translate",
                "source_language": detected_source,
                "target_language": target_language,
                "original_text": input_text
            }
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return {
                "output": f"Error translating text: {str(e)}",
                "error": str(e),
                "agent_type": "google_translate",
                "original_text": input_text
            }

    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """Get a dictionary of supported language codes and names"""
        return {
            "af": "Afrikaans",
            "sq": "Albanian",
            "am": "Amharic",
            "ar": "Arabic",
            "hy": "Armenian",
            "az": "Azerbaijani",
            "eu": "Basque",
            "be": "Belarusian",
            "bn": "Bengali",
            "bs": "Bosnian",
            "bg": "Bulgarian",
            "ca": "Catalan",
            "ceb": "Cebuano",
            "ny": "Chichewa",
            "zh-CN": "Chinese (Simplified)",
            "zh-TW": "Chinese (Traditional)",
            "co": "Corsican",
            "hr": "Croatian",
            "cs": "Czech",
            "da": "Danish",
            "nl": "Dutch",
            "en": "English",
            "eo": "Esperanto",
            "et": "Estonian",
            "tl": "Filipino",
            "fi": "Finnish",
            "fr": "French",
            "fy": "Frisian",
            "gl": "Galician",
            "ka": "Georgian",
            "de": "German",
            "el": "Greek",
            "gu": "Gujarati",
            "ht": "Haitian Creole",
            "ha": "Hausa",
            "haw": "Hawaiian",
            "iw": "Hebrew",
            "hi": "Hindi",
            "hmn": "Hmong",
            "hu": "Hungarian",
            "is": "Icelandic",
            "ig": "Igbo",
            "id": "Indonesian",
            "ga": "Irish",
            "it": "Italian",
            "ja": "Japanese",
            "jw": "Javanese",
            "kn": "Kannada",
            "kk": "Kazakh",
            "km": "Khmer",
            "ko": "Korean",
            "ku": "Kurdish (Kurmanji)",
            "ky": "Kyrgyz",
            "lo": "Lao",
            "la": "Latin",
            "lv": "Latvian",
            "lt": "Lithuanian",
            "lb": "Luxembourgish",
            "mk": "Macedonian",
            "mg": "Malagasy",
            "ms": "Malay",
            "ml": "Malayalam",
            "mt": "Maltese",
            "mi": "Maori",
            "mr": "Marathi",
            "mn": "Mongolian",
            "my": "Myanmar (Burmese)",
            "ne": "Nepali",
            "no": "Norwegian",
            "ps": "Pashto",
            "fa": "Persian",
            "pl": "Polish",
            "pt": "Portuguese",
            "pa": "Punjabi",
            "ro": "Romanian",
            "ru": "Russian",
            "sm": "Samoan",
            "gd": "Scots Gaelic",
            "sr": "Serbian",
            "st": "Sesotho",
            "sn": "Shona",
            "sd": "Sindhi",
            "si": "Sinhala",
            "sk": "Slovak",
            "sl": "Slovenian",
            "so": "Somali",
            "es": "Spanish",
            "su": "Sundanese",
            "sw": "Swahili",
            "sv": "Swedish",
            "tg": "Tajik",
            "ta": "Tamil",
            "te": "Telugu",
            "th": "Thai",
            "tr": "Turkish",
            "uk": "Ukrainian",
            "ur": "Urdu",
            "uz": "Uzbek",
            "vi": "Vietnamese",
            "cy": "Welsh",
            "xh": "Xhosa",
            "yi": "Yiddish",
            "yo": "Yoruba",
            "zu": "Zulu"
        }