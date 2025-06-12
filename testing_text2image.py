


from openai import OpenAI
import base64

client = OpenAI(api_key="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")

response = client.responses.create(
    model="gpt-4.0-mini",
    input="Generate an image of gray tabby cat hugging an otter with an orange scarf",
    tools=[{"type": "image_generation"}],
)


image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]

if image_data:
    image_base64 = image_data[0]
    with open("cat_and_otter.png", "wb") as f:
        f.write(base64.b64decode(image_base64))