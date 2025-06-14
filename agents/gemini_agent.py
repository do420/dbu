import datetime
import google.generativeai as genai
from typing import Dict, Any
from .base import BaseAgent
import logging

logger = logging.getLogger(__name__)

class GeminiAgent(BaseAgent):
    """Agent that uses Google's Gemini model
    
    Supported config parameters:
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
            
            # Add response_schema if provided (required for application/json and text/x.enum)
            mime_type = self.config.get("response_mime_type", "text/plain")
            
            if mime_type in ["application/json", "text/x.enum"]:
                if "response_schema" in self.config and self.config["response_schema"]:
                    schema = self.config["response_schema"]
                    # Validate that response_schema is a proper dict
                    if isinstance(schema, dict) and schema:
                        # Use the schema directly without cleaning
                        generation_config["response_schema"] = schema
                        logger.debug(f"Added response_schema for {mime_type}: {schema}")
                    else:
                        logger.warning(f"Invalid response_schema format, ignoring: {schema}")
                elif mime_type == "text/x.enum":
                    # For ENUM, provide a default schema if none specified
                    default_enum_schema = {
                        "type": "string",
                        "enum": ["option1", "option2", "option3"]
                    }
                    generation_config["response_schema"] = default_enum_schema
                    logger.warning(f"No response_schema provided for text/x.enum, using default: {default_enum_schema}")
                elif mime_type == "application/json":
                    # For JSON, provide a default schema if none specified
                    default_json_schema = {
                        "type": "object",
                        "properties": {
                            "response": {"type": "string"}
                        }
                    }
                    generation_config["response_schema"] = default_json_schema
                    logger.warning(f"No response_schema provided for application/json, using default: {default_json_schema}")
            
            # Remove None values and validate config
            generation_config = {k: v for k, v in generation_config.items() if v is not None}
            
            # Additional validation
            if "temperature" in generation_config:
                temp = generation_config["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    logger.warning(f"Invalid temperature: {temp}, using default")
                    generation_config["temperature"] = 0.7
            
            logger.debug(f"Using generation config: {generation_config}")
            
            # Format the messages properly for Gemini API
            formatted_input = f"[{system_message['role']}]: {system_message['content']}\n\n[{user_message['role']}]: {user_message['content']}"
            
            # Make API calls with generation config
            if context.get("history"):
                chat = self.model.start_chat(history=context["history"])
                response = chat.send_message(formatted_input, generation_config=generation_config)
            else:
                response = self.model.generate_content(formatted_input, generation_config=generation_config)
            
            # Safely extract response text
            response_text = ""
            try:
                # Method 1: Try accessing candidates directly (most reliable)
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            response_text = candidate.content.parts[0].text or ""
                            logger.debug(f"Extracted text from candidates: {len(response_text)} chars")
                        else:
                            logger.warning("Candidate content has no parts")
                    else:
                        logger.warning("Candidate has no content")
                        # Check if candidate was blocked
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = candidate.finish_reason
                            if finish_reason and finish_reason != "STOP":
                                response_text = f"Content generation stopped: {finish_reason}"
                
                # Method 2: Try direct text access as fallback
                elif hasattr(response, 'text'):
                    try:
                        response_text = response.text or ""
                        logger.debug(f"Extracted text via response.text: {len(response_text)} chars")
                    except NotImplementedError:
                        logger.warning("response.text raised NotImplementedError")
                        response_text = "Response text not available (NotImplementedError)"
                    except Exception as text_err:
                        logger.warning(f"Error accessing response.text: {text_err}")
                        response_text = f"Error accessing response text: {str(text_err)}"
                
                # If still no text, check for blocking
                if not response_text:
                    if hasattr(response, 'prompt_feedback'):
                        feedback = response.prompt_feedback
                        if hasattr(feedback, 'block_reason') and feedback.block_reason:
                            response_text = f"Content blocked: {feedback.block_reason}"
                        else:
                            response_text = "No text content returned from Gemini API"
                    else:
                        response_text = "Empty response from Gemini API"
                        
            except Exception as extraction_error:
                logger.error(f"Error during text extraction: {extraction_error}")
                response_text = f"Failed to extract response: {str(extraction_error)}"
            
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
                        "completion_tokens": len(response_text.split()) * 1.3 if response_text else 0,  # Rough estimate
                        "total_tokens": len(formatted_input.split()) * 1.3 + (len(response_text.split()) * 1.3 if response_text else 0)  # Rough estimate
                    }
            except Exception as e:
                logger.warning(f"Error getting token usage: {e}, using estimation")
                token_usage = {
                    "prompt_tokens": len(formatted_input.split()) * 1.3,  # Rough estimate
                    "completion_tokens": len(response_text.split()) * 1.3 if response_text else 0,  # Rough estimate
                    "total_tokens": len(formatted_input.split()) * 1.3 + (len(response_text.split()) * 1.3 if response_text else 0)  # Rough estimate
                }
            
            logger.debug(f"GeminiAgent response received successfully")
            return {
                "output": response_text,
                "raw_response": str(response),
                "agent_type": "gemini",
                "token_usage": token_usage
            }
        except Exception as e:
            error_message = str(e)
            exception_type = type(e).__name__
            logger.error(f"Error with Gemini API: {error_message}")
            logger.error(f"Exception type: {exception_type}")
            
            # More specific error messages
            if exception_type == "NotImplementedError":
                error_message = "Gemini API response format issue - content may have been blocked"
            elif "API_KEY_INVALID" in error_message:
                error_message = "Invalid Gemini API key"
            elif "QUOTA_EXCEEDED" in error_message:
                error_message = "Gemini API quota exceeded"
            elif "PERMISSION_DENIED" in error_message:
                error_message = "Permission denied for Gemini API"
            elif "response_schema" in error_message or "schema" in error_message.lower():
                error_message = "Invalid response schema configuration - check schema format and remove unsupported fields"
            elif not error_message.strip():
                error_message = "Unknown Gemini API error"
            
            return {
                "output": f"Error with Gemini API: {error_message}",
                "error": error_message,
                "agent_type": "gemini",
                "token_usage": {"total_tokens": 0}
            }
    
