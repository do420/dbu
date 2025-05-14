import requests
from typing import Dict, Any
from .base import BaseAgent

class CustomEndpointAgent(BaseAgent):
    """Agent that sends requests to a user-defined HTTP endpoint."""

    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        endpoint_url = self.config.get("endpoint_url")
        if not endpoint_url:
            return {"error": "No endpoint_url configured in agent.", "output": "No endpoint_url configured."}

        # Prepare payload
        payload = {
            "input": input_data,
            "context": context
        }

        try:
            resp = requests.post(endpoint_url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {"error": f"Failed to call external endpoint: {str(e)}", "output": f"Failed to call external endpoint: {str(e)}"}

        # Validate response format
        if not isinstance(data, dict):
            return {"error": "Invalid response: not a JSON object", "output": "Invalid response: not a JSON object"}

        if "output_type" not in data or "output" not in data:
            return {"error": "Invalid response: missing required fields", "output": "Invalid response: missing required fields"}

        if data["output_type"] not in ["text", "image", "sound"]:
            return {"error": "Invalid output_type in response", "output": "Invalid output_type in response"}

        return {
            "output_type": data["output_type"],
            "output": data["output"],
            "meta": data.get("meta", {}),
            "agent_type": "custom_endpoint"
        }