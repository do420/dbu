#!/usr/bin/env python3

"""
Test script to verify the checklist approach in the chat-generate endpoint
"""

import json

def test_checklist_scenarios():
    """Test different conversation scenarios with the checklist approach"""
    
    test_scenarios = [
        {
            "name": "Scenario 1: Just starting - no info gathered",
            "conversation_history": [],
            "message": "Hi, I want to create a service",
            "expected_behavior": "Should ask about service purpose"
        },
        {
            "name": "Scenario 2: Purpose gathered, missing input type",
            "conversation_history": [
                {"role": "user", "content": "Hi, I want to create a service"},
                {"role": "assistant", "content": "Great! What specific task would you like your service to accomplish?"},
                {"role": "user", "content": "I want it to generate funny jokes"}
            ],
            "message": "Make it really creative",
            "expected_behavior": "Should ask about input type, mention joke generation progress"
        },
        {
            "name": "Scenario 3: Purpose + Input gathered, missing output type",
            "conversation_history": [
                {"role": "user", "content": "Hi, I want to create a service"},
                {"role": "assistant", "content": "Great! What specific task would you like your service to accomplish?"},
                {"role": "user", "content": "I want it to generate funny jokes"},
                {"role": "assistant", "content": "Perfect! I understand you want to generate funny jokes. What type of input will you provide - text, image, or audio?"},
                {"role": "user", "content": "Text input"}
            ],
            "message": "Yes, text",
            "expected_behavior": "Should ask about output type, mention joke generation + text input progress"
        },
        {
            "name": "Scenario 4: Purpose + Input + Output gathered, missing name",
            "conversation_history": [
                {"role": "user", "content": "Hi, I want to create a service"},
                {"role": "assistant", "content": "Great! What specific task would you like your service to accomplish?"},
                {"role": "user", "content": "I want it to generate funny jokes"},
                {"role": "assistant", "content": "Perfect! I understand you want to generate funny jokes. What type of input will you provide - text, image, or audio?"},
                {"role": "user", "content": "Text input"},
                {"role": "assistant", "content": "Excellent! So far I know: generate funny jokes with text input. What would you like to get back - text, image, or audio?"},
                {"role": "user", "content": "Text output with jokes"}
            ],
            "message": "Yes, just text",
            "expected_behavior": "Should ask for service name, mention progress: jokes, text→text"
        },
        {
            "name": "Scenario 5: All info gathered, should auto-create",
            "conversation_history": [
                {"role": "user", "content": "Hi, I want to create a service"},
                {"role": "assistant", "content": "Great! What specific task would you like your service to accomplish?"},
                {"role": "user", "content": "I want it to generate funny jokes"},
                {"role": "assistant", "content": "Perfect! I understand you want to generate funny jokes. What type of input will you provide - text, image, or audio?"},
                {"role": "user", "content": "Text input"},
                {"role": "assistant", "content": "Excellent! So far I know: generate funny jokes with text input. What would you like to get back - text, image, or audio?"},
                {"role": "user", "content": "Text output"},
                {"role": "assistant", "content": "Great! I have: generate funny jokes, text → text. What would you like to name this service?"},
                {"role": "user", "content": "Joke Generator"}
            ],
            "message": "That's perfect",
            "expected_behavior": "Should automatically create service with CREATE_SERVICE: JSON"
        },
        {
            "name": "Scenario 6: All info in one message",
            "conversation_history": [],
            "message": "Create a translation service that takes English text and returns Spanish text, call it 'English to Spanish Translator'",
            "expected_behavior": "Should automatically create service since all checklist items are covered"
        }
    ]
    
    print("CHECKLIST APPROACH TEST SCENARIOS")
    print("=" * 50)
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        print("-" * len(scenario['name']))
        print(f"Message: '{scenario['message']}'")
        print(f"Conversation history: {len(scenario['conversation_history'])} messages")
        print(f"Expected: {scenario['expected_behavior']}")
        
        # Analyze what info is available in conversation
        gathered_info = analyze_conversation_for_checklist(scenario['conversation_history'], scenario['message'])
        print(f"Analysis:")
        for item, status in gathered_info.items():
            checkmark = "✓" if status['complete'] else "☐"
            print(f"  {checkmark} {item}: {status['value'] if status['complete'] else 'Missing'}")

def analyze_conversation_for_checklist(conversation_history, current_message):
    """Analyze conversation to determine checklist completion"""
    
    all_text = " ".join([msg['content'].lower() for msg in conversation_history] + [current_message.lower()])
    
    checklist = {
        "SERVICE PURPOSE": {"complete": False, "value": None},
        "INPUT TYPE": {"complete": False, "value": None},
        "OUTPUT TYPE": {"complete": False, "value": None},
        "SERVICE NAME": {"complete": False, "value": None},
        "AGENT SELECTION": {"complete": False, "value": None}
    }
    
    # Check for purpose indicators
    purpose_keywords = ['generate', 'translate', 'convert', 'analyze', 'create', 'joke', 'story', 'summary']
    for keyword in purpose_keywords:
        if keyword in all_text:
            checklist["SERVICE PURPOSE"]["complete"] = True
            checklist["SERVICE PURPOSE"]["value"] = f"Contains '{keyword}'"
            break
    
    # Check for input type
    if any(word in all_text for word in ['text input', 'text', 'typing', 'write']):
        checklist["INPUT TYPE"]["complete"] = True
        checklist["INPUT TYPE"]["value"] = "text"
    elif any(word in all_text for word in ['image', 'picture', 'photo']):
        checklist["INPUT TYPE"]["complete"] = True
        checklist["INPUT TYPE"]["value"] = "image"
    elif any(word in all_text for word in ['audio', 'sound', 'voice', 'speak']):
        checklist["INPUT TYPE"]["complete"] = True
        checklist["INPUT TYPE"]["value"] = "sound"
    
    # Check for output type
    if any(word in all_text for word in ['text output', 'text', 'writing', 'written']):
        checklist["OUTPUT TYPE"]["complete"] = True
        checklist["OUTPUT TYPE"]["value"] = "text"
    elif any(word in all_text for word in ['image', 'picture', 'photo']):
        checklist["OUTPUT TYPE"]["complete"] = True
        checklist["OUTPUT TYPE"]["value"] = "image"
    elif any(word in all_text for word in ['audio', 'sound', 'voice', 'speech']):
        checklist["OUTPUT TYPE"]["complete"] = True
        checklist["OUTPUT TYPE"]["value"] = "sound"
    
    # Check for service name
    name_indicators = ['call it', 'name it', 'generator', 'translator', 'service']
    for indicator in name_indicators:
        if indicator in all_text:
            checklist["SERVICE NAME"]["complete"] = True
            checklist["SERVICE NAME"]["value"] = f"Contains '{indicator}'"
            break
    
    # Check if agent selection would be automatic
    if checklist["SERVICE PURPOSE"]["complete"] and checklist["INPUT TYPE"]["complete"] and checklist["OUTPUT TYPE"]["complete"]:
        checklist["AGENT SELECTION"]["complete"] = True
        checklist["AGENT SELECTION"]["value"] = "Auto-selectable"
    
    return checklist

if __name__ == "__main__":
    test_checklist_scenarios()
