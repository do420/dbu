import json
import os

PRICING_FILE = os.path.join(os.path.dirname(__file__), 'gemini_pricing.json')

class pricing_calculator:
    @staticmethod
    def _load_pricing():
        with open(PRICING_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['models']

    @staticmethod
    def _find_model(model_name, models):
        for model in models:
            if model_name.lower() in model['name'].lower():
                return model
        return None

    @staticmethod
    def add_pricing_to_response(response: dict, model_name: str) -> dict:
        models = pricing_calculator._load_pricing()
        model = pricing_calculator._find_model(model_name, models)
        if not model or 'token_usage' not in response:
            response['pricing'] = None
            return response

        total_tokens = response['token_usage'].get('total_tokens', 0)
        # Always use paid tier, under_128k_tokens if available, else lowest available
        input_price = None
        output_price = None
        # Try to get under_128k_tokens or under_200k_tokens pricing if available
        for key in ['under_128k_tokens', 'under_200k_tokens', 'text_image_video', 'paid']:
            if 'input_price' in model and 'paid' in model['input_price']:
                if isinstance(model['input_price']['paid'], dict) and key in model['input_price']['paid']:
                    input_price = model['input_price']['paid'][key]
                    break
                elif isinstance(model['input_price']['paid'], (int, float)):
                    input_price = model['input_price']['paid']
                    break
            elif 'input_price' in model and isinstance(model['input_price'], dict) and key in model['input_price']:
                input_price = model['input_price'][key]
                break
        for key in ['under_128k_tokens', 'under_200k_tokens', 'non_thoughtful', 'paid']:
            if 'output_price' in model and 'paid' in model['output_price']:
                if isinstance(model['output_price']['paid'], dict) and key in model['output_price']['paid']:
                    output_price = model['output_price']['paid'][key]
                    break
                elif isinstance(model['output_price']['paid'], (int, float)):
                    output_price = model['output_price']['paid']
                    break
            elif 'output_price' in model and isinstance(model['output_price'], dict) and key in model['output_price']:
                output_price = model['output_price'][key]
                break
        # If image generation, try image_price or image_generation_price
        if not input_price and 'image_generation_price' in model and 'paid' in model['image_generation_price']:
            input_price = model['image_generation_price']['paid']
        if not input_price and 'image_price' in model and 'paid' in model['image_price']:
            input_price = model['image_price']['paid']
        # Calculate price (per 1M tokens)
        price = 0.0
        if input_price:
            price += (total_tokens / 1_000_000) * float(input_price)
        if output_price:
            price += (total_tokens / 1_000_000) * float(output_price)
        response['pricing'] = {
            'model': model.get('name', model_name),
            'input_price_per_million': input_price,
            'output_price_per_million': output_price,
            'total_tokens': total_tokens,
            'estimated_cost_usd': round(price, 6)
        }
        return response
