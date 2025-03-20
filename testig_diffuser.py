from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion model
def load_model():
    # Use a correct model path from Hugging Face
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    model.to("cuda")  # Use "cpu" if no GPU is available
    return model

# Generate image from text prompt
def generate_image(prompt, model):
    image = model(prompt).images[0]  # Generate image
    return image

def main():
    # Load model once
    model = load_model()

    # Input your prompt
    prompt = input("Enter your prompt: ")

    # Generate the image
    generated_image = generate_image(prompt, model)

    # Save and show the generated image
    generated_image.save("generated_image.png")
    generated_image.show()

if __name__ == "__main__":
    main()
