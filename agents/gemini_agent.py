import datetime
import google.generativeai as genai
from typing import Dict, Any
from .base import BaseAgent
import logging

logger = logging.getLogger(__name__)

class GeminiAgent(BaseAgent):
    """Agent that uses Google's Gemini model"""
    
    def __init__(self, config: Dict[str, Any], system_instruction: str):
        super().__init__(config, system_instruction)
        # Initialize the Gemini API
        genai.configure(api_key=config.get("api_key"))
        self.model = genai.GenerativeModel(config.get("model_name", "gemini-1.5-flash"))
    
    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input text using Gemini and return response"""
        context = context or {}
        
        try:
            logger.debug(f"GeminiAgent processing input: {input_text} with context: {context}")
            

            # Use system instruction from the agent and include today's date
            today_str = datetime.datetime.now().strftime("%B %d, %Y")
            date_info = f"If applicable, today's date is: {today_str}."
            
            if self.system_instruction:
                system_prompt = f"{self.system_instruction}\n\n{date_info}"
            else:
                system_prompt = date_info



            logger.debug(f"Using system prompt: {system_prompt}")
            
            # Prepare generation config
            generation_config = {}
            
            if system_prompt:
                formatted_input = f"System: {system_prompt}\n\nUser: {input_text}"
            else:
                formatted_input = input_text
            
            # Make API calls
            if context.get("history"):
                chat = self.model.start_chat(history=context["history"])
                
                if generation_config:
                    response = chat.send_message(formatted_input, generation_config=generation_config)
                else:
                    response = chat.send_message(formatted_input)
            else:
                if generation_config:
                    response = self.model.generate_content(formatted_input, generation_config=generation_config)
                else:
                    response = self.model.generate_content(formatted_input)
            
            # Track token usage (approximate since Gemini doesn't provide exact tokens)
            # This is an estimation - in a real system you would use the actual count from the API
            token_usage = {
                "prompt_tokens": len(formatted_input.split()) * 1.3,  # Rough estimate
                "completion_tokens": len(response.text.split()) * 1.3,  # Rough estimate
                "total_tokens": len(formatted_input.split() + response.text.split()) * 1.3  # Rough estimate
            }
            
            logger.debug(f"GeminiAgent response received")
            return {
                "output": response.text,
                "raw_response": str(response),
                "agent_type": "gemini",
                "token_usage": token_usage
            }
        except Exception as e:
            logger.error(f"Error with Gemini API: {str(e)}")
            return {
                "output": f"Error with Gemini API: {str(e)}",
                "error": str(e),
                "agent_type": "gemini",
                "token_usage": {"total_tokens": 0}
            }