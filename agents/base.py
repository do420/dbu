from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """Base abstract class for all agents"""
    
    def __init__(self, config: Dict[str, Any], system_instruction: str):
        self.config = config
        self.system_instruction = system_instruction
    
    @abstractmethod
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input and return a response
        
        Args:
            input_data: The input data to process (text, image path, etc.)
            context: Optional context dict that may contain:
                - history: Previous conversation history
                - step_results: Results from previous steps in a multi-agent workflow
                
        Returns:
            Dict with at least an "output" key containing the response
        """
        pass