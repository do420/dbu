from typing import Dict, Any

from .transribe_agent import TranscribeAgent

from .gemini_agent import GeminiAgent
from .openai_agent import OpenAIAgent

from .tts_agent import TTSAgent
from .bark_agent import BarkTTSAgent

from .gemini_image_generation_agent import GeminiImageGeneration
from .internet_research_agent import InternetResearchAgent
from .document_parser_agent import DocumentParserAgent
from .custom_endpoint_agent import CustomEndpointAgent
from .google_translate_agent import GoogleTranslateAgent
import logging

logger = logging.getLogger(__name__)

def create_agent(agent_type: str, config: Dict[str, Any], system_instruction: str):
    """Factory function to create appropriate agent based on type"""
    logger.debug(f"Creating agent of type {agent_type} with config {config}")
    
    if agent_type.lower() == "gemini":
        return GeminiAgent(config, system_instruction)
    elif agent_type.lower() == "openai":
        return OpenAIAgent(config, system_instruction)
    elif agent_type.lower() == "edge_tts":
        return TTSAgent(config, system_instruction)
    elif agent_type.lower() == "bark_tts":
        return BarkTTSAgent(config, system_instruction)
    elif agent_type.lower() == "transcribe":
        return TranscribeAgent(config, system_instruction)
    elif agent_type.lower() == "gemini_text2image":
        return GeminiImageGeneration(config, system_instruction)
    elif agent_type.lower() == "internet_research":
        return InternetResearchAgent(config, system_instruction)
    elif agent_type.lower() == "document_parser":
        return DocumentParserAgent(config, system_instruction)
    elif agent_type.lower() == "custom_endpoint":
        return CustomEndpointAgent(config, system_instruction)
    elif agent_type.lower() == "google_translate":
        return GoogleTranslateAgent(config, system_instruction)
    else:
        logger.error(f"Unknown agent type: {agent_type}")
        raise ValueError(f"Unknown agent type: {agent_type}")