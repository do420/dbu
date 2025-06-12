import datetime
import google.generativeai as genai
from typing import Dict, Any
from .base import BaseAgent
import logging

logger = logging.getLogger(__name__)

class GeminiAgent(BaseAgent):
    """Agent that uses Google's Gemini model
    
    Supported config parameters:
    - api_key: Gemini API key (required)
    - model_name: Model name (default: "gemini-1.5-flash")
    - temperature: Controls randomness (0.0-2.0, default: 0.7)
    - top_p: Controls diversity via nucleus sampling (0.0-1.0, default: 0.8)
    - top_k: Controls diversity via top-k sampling (1-40, default: 40)
    - max_output_tokens: Maximum tokens in response (default: 8192)
    - response_mime_type: Response format (default: "text/plain")
        Available options:
        * "text/plain" - Standard text output (default)
        * "application/json" - JSON response format
        * "text/x.enum" - ENUM as string response
    - response_schema: Output schema (optional, requires "application/json" mime type)
    """
    
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
              # Prepare generation config with parameters from config or defaults
            generation_config = {
                "temperature": self.config.get("temperature", 0.7),
                "top_p": self.config.get("top_p", 0.8),
                "top_k": self.config.get("top_k", 40),
                "max_output_tokens": self.config.get("max_output_tokens", 8192),
                "response_mime_type": self.config.get("response_mime_type", "text/plain")
            }
            
            # Add response_schema if provided (only valid with application/json mime type)
            if "response_schema" in self.config:
                generation_config["response_schema"] = self.config["response_schema"]
            
            # Remove None values and convert to GenerationConfig if needed
            generation_config = {k: v for k, v in generation_config.items() if v is not None}
            
            logger.debug(f"Using generation config: {generation_config}")
            
            # Format the messages properly for Gemini API
            formatted_input = f"[{system_message['role']}]: {system_message['content']}\n\n[{user_message['role']}]: {user_message['content']}"
            
            # Make API calls with generation config
            if context.get("history"):
                chat = self.model.start_chat(history=context["history"])
                response = chat.send_message(formatted_input, generation_config=generation_config)
            else:
                response = self.model.generate_content(formatted_input, generation_config=generation_config)
            
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