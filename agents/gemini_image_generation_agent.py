import google.generativeai as genai
from typing import Dict, Any, List
from .base import BaseAgent
import logging
from PIL import Image
import io
import base64 # Import the base64 module
logger = logging.getLogger(__name__)

class GeminiImageGeneration(BaseAgent):
    """Agent that uses Google's Gemini model for image generation."""

    def __init__(self, config: Dict[str, Any], system_instruction: str = "Generate an image based on the following description."):
        super().__init__(config, system_instruction)
        genai.configure(api_key=config.get("api_key"))
        self.model = genai.GenerativeModel(config.get("model_name", "gemini-pro-vision")) # Use a vision model

    async def process(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process the image generation prompt using Gemini and return the image data."""
        context = context or {}
        try:
            logger.debug(f"GeminiImageGeneration processing prompt: {prompt} with context: {context}")

            # Prepare the prompt with system instructions
            formatted_prompt = f"System: {self.system_instruction}\n\nUser: {prompt}" if self.system_instruction else prompt

            # Make the API call for image generation
            response = self.model.generate_content([formatted_prompt])
            response.resolve() # Ensure the response is fully loaded

            if response.parts and hasattr(response.parts[0], "data"):
                image_data = response.parts[0].data
                image = Image.open(io.BytesIO(image_data))
                # Convert image to bytes (e.g., PNG format) for consistent handling
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="PNG")
                image_bytes.seek(0)
                image_base64 = base64.b64encode(image_bytes.read()).decode("utf-8")

                logger.debug(f"GeminiImageGeneration generated an image.")
                return {
                    "output": image_base64, # Return as base64 encoded string
                    "raw_response": str(response),
                    "agent_type": "gemini-image",
                    "mime_type": "image/png"
                }
            else:
                logger.warning(f"GeminiImageGeneration did not return image data. Response: {response}")
                return {
                    "output": "Error: No image data received from Gemini.",
                    "error": "No image data in response",
                    "raw_response": str(response),
                    "agent_type": "gemini-image",
                    "token_usage": {"total_tokens": 0} # No clear token usage for image generation
                }

        except Exception as e:
            logger.error(f"Error with Gemini Image Generation API: {str(e)}")
            return {
                "output": f"Error with Gemini Image Generation API: {str(e)}",
                "error": str(e),
                "agent_type": "gemini-image",
                "token_usage": {"total_tokens": 0}
            }

