from diffusers import StableDiffusionPipeline
import torch  

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")  
pipe.enable_attention_slicing()  

def generate_image(prompt, filename="output.png"):  
    image = pipe(prompt).images[0]  
    image.save(filename)  

generate_image("A cyberpunk city at night")  
