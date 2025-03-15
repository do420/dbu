from typing import Dict, Any
import openai
from .base import BaseAgent
import logging

logger = logging.getLogger(__name__)

class OpenAIAgent(BaseAgent):
    """Agent that uses OpenAI's models"""
    
    def __init__(self, config: Dict[str, Any], system_instruction: str):
        super().__init__(config, system_instruction)
        # Initialize OpenAI API client
        self.client = openai.OpenAI(api_key=config.get("api_key"))
        self.model = config.get("model_name", "gpt-3.5-turbo")
    
    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input text using OpenAI and return response"""
        context = context or {}
        
        try:
            logger.debug(f"OpenAIAgent processing input: {input_text} with context: {context}")
            
            # Get system prompt from agent
            system_prompt = self.system_instruction
            
            # Create messages for the conversation
            messages = []
            
            # Add system message if system prompt exists
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add previous conversation history if available
            if context.get("history"):
                messages.extend(context["history"])
            
            # Add current user message
            messages.append({"role": "user", "content": input_text})
            
            logger.debug(f"OpenAI messages: {messages}")
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            response_text = response.choices[0].message.content
            logger.debug(f"OpenAIAgent response: {response_text}")
            
            # Get token usage from response
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return {
                "output": response_text,
                "raw_response": str(response),
                "agent_type": "openai",
                "token_usage": token_usage
            }
        except Exception as e:
            logger.error(f"Error with OpenAI API: {str(e)}")
            return {
                "output": f"Error with OpenAI API: {str(e)}",
                "error": str(e),
                "agent_type": "openai",
                "token_usage": {"total_tokens": 0}
            }