from typing import Dict, Any, List
from .base import BaseAgent
import logging

logger = logging.getLogger(__name__)

class WorkflowProcessor:
    """
    Processes a workflow of multiple agents
    """
    
    def __init__(self, agents: Dict[str, BaseAgent], workflow: Dict[str, Any]):
        """
        Initialize with a map of agent IDs to agent instances and a workflow definition
        """
        self.agents = agents
        self.workflow = workflow
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input through the workflow
        """
        logger.debug(f"Starting workflow process with input: {input_data}")
        context = context or {}
        
        # Get the start node - use the specified start_node or default to "0"
        current_step = self.workflow.get("start_node", "0")
        current_input = input_data
        results = []
        total_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        while current_step is not None:
            # Get node configuration
            step_config = self.workflow["nodes"].get(str(current_step))
            if not step_config:
                logger.error(f"No step configuration found for step: {current_step}")
                break
                
            agent_id = step_config.get("agent_id")
            if agent_id is None:
                logger.error(f"No agent ID specified for step: {current_step}")
                results.append({
                    "step": current_step,
                    "error": f"No agent ID specified for step: {current_step}"
                })
                break
            
            agent = self.agents.get(str(agent_id))
            
            if not agent:
                logger.error(f"Agent {agent_id} not found for step: {current_step}")
                results.append({
                    "step": current_step,
                    "agent_id": agent_id,
                    "input": current_input,
                    "error": f"Agent {agent_id} not found"
                })
                break
            
            logger.debug(f"Processing step {current_step} with agent {agent_id}")
            
            try:
                # Process with the current agent
                agent_result = await agent.process(current_input, context)
                
                # Store the result
                result = {
                    "step": current_step,
                    "agent_id": agent_id,
                    "input": current_input,
                    "output": agent_result.get("output"),
                    "raw": agent_result
                }
                results.append(result)
                
                # Update context with the current result
                if "step_results" not in context:
                    context["step_results"] = {}
                context["step_results"][str(current_step)] = agent_result.get("output")
                
                # Update token usage
                if "token_usage" in agent_result:
                    for key, value in agent_result["token_usage"].items():
                        if key in total_token_usage:
                            total_token_usage[key] += value
                
            except Exception as e:
                logger.error(f"Error processing with agent {agent_id}: {str(e)}")
                results.append({
                    "step": current_step,
                    "agent_id": agent_id,
                    "input": current_input,
                    "error": str(e)
                })
                break
            
            # Get the next step directly from the "next" field
            next_step = step_config.get("next")
            
            # If next_step is None, we're done
            if next_step is None:
                break
                
            # Update the current step and input
            current_step = str(next_step)  # Ensure it's a string for consistency
            current_input = agent_result.get("output")
        
        # Safely check for final output
        final_output = None
        if results:
            last_result = results[-1]
            final_output = last_result.get("output") if "output" in last_result else None
            
        return {
            "results": results,
            "final_output": final_output,
            "token_usage": total_token_usage
        }