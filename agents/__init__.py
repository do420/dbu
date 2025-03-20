from typing import Dict, Any

from .transribe_agent import TranscribeAgent
from .gemini_agent import GeminiAgent
from .openai_agent import OpenAIAgent
from .tts_agent import TTSAgent
from .bark_agent import BarkTTSAgent

import logging

logger = logging.getLogger(__name__)

def create_agent(agent_type: str, config: Dict[str, Any], system_instruction: str):
    """Factory function to create appropriate agent based on type"""
    logger.debug(f"Creating agent of type {agent_type} with config {config}")
    
    if agent_type.lower() == "gemini":
        return GeminiAgent(config, system_instruction)
    elif agent_type.lower() == "openai":
        return OpenAIAgent(config, system_instruction)
    elif agent_type.lower() == "text2speech" or agent_type.lower() == "tts":
        return TTSAgent(config, system_instruction)
    elif agent_type.lower() == "bark_tts" or agent_type.lower() == "suno-bark":
        return BarkTTSAgent(config, system_instruction)
    elif agent_type.lower() == "transcribe" or agent_type.lower() == "whisperx":
        return TranscribeAgent(config, system_instruction)
    else:
        logger.error(f"Unknown agent type: {agent_type}")
        raise ValueError(f"Unknown agent type: {agent_type}")