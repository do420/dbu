#!/usr/bin/env python3

"""
Test script to simulate Gemini CREATE_SERVICE response parsing
"""

import json

def test_json_parsing():
    """Test the JSON parsing logic"""
    
    # Simulate various Gemini response formats
    test_responses = [
        # Format 1: With markdown
        '''CREATE_SERVICE: ```json
{
  "service": {
    "name": "Joke Generator",
    "description": "Generates funny jokes based on text input",
    "input_type": "text",
    "output_type": "text",
    "is_public": false
  },
  "agents": [
    {
      "name": "Joke Creator",
      "agent_type": "gemini",
      "system_instruction": "Generate funny, clean jokes based on the input topic or theme",
      "config": {"temperature": 0.8, "model": "gemini-1.5-flash"},
      "input_type": "text",
      "output_type": "text"
    }
  ],
  "workflow": {
    "nodes": {
      "0": {
        "agent_id": 0,
        "next": null
      }
    }
  }
}
```''',
        
        # Format 2: Without markdown
        '''CREATE_SERVICE: {
  "service": {
    "name": "Joke Generator",
    "description": "Generates funny jokes based on text input",
    "input_type": "text",
    "output_type": "text",
    "is_public": false
  },
  "agents": [
    {
      "name": "Joke Creator",
      "agent_type": "gemini",
      "system_instruction": "Generate funny, clean jokes based on the input topic or theme",
      "config": {"temperature": 0.8, "model": "gemini-1.5-flash"},
      "input_type": "text",
      "output_type": "text"
    }
  ],
  "workflow": {
    "nodes": {
      "0": {
        "agent_id": 0,
        "next": null
      }
    }
  }
}'''
    ]
    
    for i, response_text in enumerate(test_responses, 1):
        print(f"\n--- Testing Format {i} ---")
        print(f"Original response:\n{response_text}")
        
        # Simulate the parsing logic from mini_services.py
        if response_text.startswith("CREATE_SERVICE:"):
            json_text = response_text[15:].strip()  # Remove "CREATE_SERVICE:" prefix
            
            # Remove markdown code block formatting if present
            if json_text.startswith("```json"):
                json_text = json_text[7:]  # Remove ```json
            if json_text.endswith("```"):
                json_text = json_text[:-3]  # Remove ```
            json_text = json_text.strip()
            
            print(f"\nExtracted JSON:\n{json_text}")
            
            try:
                generated_data = json.loads(json_text)
                print("✅ JSON parsing successful!")
                
                # Validate structure
                required_keys = ["service", "agents", "workflow"]
                missing_keys = [key for key in required_keys if key not in generated_data]
                
                if missing_keys:
                    print(f"❌ Missing required keys: {missing_keys}")
                else:
                    print("✅ All required keys present!")
                    
                    # Check agent types
                    for idx, agent in enumerate(generated_data["agents"]):
                        agent_type = agent.get("agent_type")
                        print(f"  Agent {idx}: type = {agent_type}")
                        
                        valid_types = [
                            "gemini", "openai", "claude", "edge_tts", "bark_tts", 
                            "transcribe", "gemini_text2image", "internet_research", 
                            "document_parser", "custom_endpoint_llm", "rag", "google_translate"
                        ]
                        
                        if agent_type in valid_types:
                            print(f"    ✅ Valid agent type")
                        else:
                            print(f"    ❌ Invalid agent type! Valid types: {valid_types}")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"Problematic JSON: {repr(json_text)}")
        else:
            print("❌ Response doesn't start with CREATE_SERVICE:")

if __name__ == "__main__":
    test_json_parsing()
