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
            
            # Create structured system prompt to prevent prompt injection
            if self.system_instruction:
                system_content = f"{self.system_instruction}\n\n{date_info}"
            else:
                system_content = date_info

            # Structure the prompt with role-based format
            system_message = {"role": "system", "content": system_content}
            user_message = {"role": "user", "content": input_text}

            logger.debug(f"Using system message: {system_message}")
            
            # Prepare generation config
            generation_config = {}
            
            # Format the messages properly for Gemini API
            formatted_input = f"[{system_message['role']}]: {system_message['content']}\n\n[{user_message['role']}]: {user_message['content']}"
            
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
            
            # Get actual token usage from Gemini response
            token_usage = {}
            try:
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    token_usage = {
                        "prompt_tokens": usage.prompt_token_count,
                        "completion_tokens": usage.candidates_token_count,
                        "total_tokens": usage.total_token_count
                    }
                    print(f"ACTUAL Token usage metadata: {token_usage}")
                    logger.debug(f"Actual token usage from Gemini: {token_usage}")
                else:
                    # Fallback to estimation if usage_metadata is not available
                    print("Usage metadata not available, using estimation")
                    logger.warning("Usage metadata not available, using estimation")
                    token_usage = {
                        "prompt_tokens": len(formatted_input.split()) * 1.3,  # Rough estimate
                        "completion_tokens": len(response.text.split()) * 1.3,  # Rough estimate
                        "total_tokens": len(formatted_input.split() + response.text.split()) * 1.3  # Rough estimate
                    }
            except Exception as e:
                logger.warning(f"Error getting token usage: {e}, using estimation")
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