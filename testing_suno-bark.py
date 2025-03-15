import torch
import numpy as np
from bark import generate_audio
from scipy.io.wavfile import write

def text_to_speech(text, filename="output.wav"):
    audio_array = generate_audio(text)
    sample_rate = 24000  # Bark uses 24kHz sample rate
    write(filename, sample_rate, (audio_array * 32767).astype(np.int16))
    print(f"Audio saved as {filename}")

text_to_speech("Yo yo yo, what up everyone! Welcome to the Bark text-to-speech demo! In this podcast, we'll be discussing the latest in AI and machine learning news. Let's get started!") 
