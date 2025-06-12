import openai
from typing import Dict, Any, List
from .base import BaseAgent
from core.pricing_utils import pricing_calculator
import logging
import requests
import base64
import io
logger = logging.getLogger(__name__)

class OpenAIImageGeneration(BaseAgent):
    """Agent that uses OpenAI's DALL-E model for image generation."""

    def __init__(self, config: Dict[str, Any], system_instruction: str = "Generate an image based on the following description."):
        super().__init__(config, system_instruction)
        self.client = openai.OpenAI(api_key=config.get("api_key"))
        self.model_name = config.get("model_name", "dall-e-3")

    async def process(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process the image generation prompt using OpenAI DALL-E and return the image data."""
        context = context or {}
        try:
            logger.debug(f"OpenAIImageGeneration processing prompt: {prompt} with context: {context}")

            # Prepare the prompt with system instructions
            formatted_prompt = f"{self.system_instruction}\n\n{prompt}" if self.system_instruction else prompt

            # Make the API call for image generation
            response = self.client.images.generate(
                model=self.model_name,
                prompt=formatted_prompt,
                n=1
            )

            if response.data and len(response.data) > 0:
                image_url = response.data[0].url
                
                # Download the image and convert to base64
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_base64 = base64.b64encode(image_response.content).decode("utf-8")
                    
                    logger.debug(f"OpenAIImageGeneration generated an image.")
                    
                    # Structure output to match the expected format
                    image_data = [
                        {
                            "result": image_base64,
                            "type": "image_generation_call"
                        }
                    ]
                    
                    result = {
                        "output": image_data,
                        "raw_response": str(response),
                        "agent_type": "openai-image",
                        "mime_type": "image/png",
                        "image_url": image_url,  # Also include the original URL
                        "token_usage": {"total_tokens": 0}
                    }
                    model_name = self.config.get("model_name", "dall-e-3")
                    result = pricing_calculator.add_pricing_to_response(result, model_name)
                    return result
                else:
                    logger.warning(f"Failed to download image from URL: {image_url}")
                    result = {
                        "output": "Error: Failed to download generated image.",
                        "error": f"Failed to download image, status code: {image_response.status_code}",
                        "raw_response": str(response),
                        "agent_type": "openai-image",
                        "token_usage": {"total_tokens": 0}
                    }
                    model_name = self.config.get("model_name", "dall-e-3")
                    result = pricing_calculator.add_pricing_to_response(result, model_name)
                    return result
            else:
                logger.warning(f"OpenAIImageGeneration did not return image data. Response: {response}")
                result = {
                    "output": "Error: No image data received from OpenAI.",
                    "error": "No image data in response",
                    "raw_response": str(response),
                    "agent_type": "openai-image",
                    "token_usage": {"total_tokens": 0}
                }
                model_name = self.config.get("model_name", "dall-e-3")
                result = pricing_calculator.add_pricing_to_response(result, model_name)
                return result

        except Exception as e:
            logger.error(f"Error with OpenAI Image Generation API: {str(e)}")
            result = {
                "output": f"Error with OpenAI Image Generation API: {str(e)}",
                "error": str(e),
                "agent_type": "openai-image",
                "token_usage": {"total_tokens": 0}
            }
            model_name = self.config.get("model_name", "dall-e-3")
            result = pricing_calculator.add_pricing_to_response(result, model_name)
            return result

