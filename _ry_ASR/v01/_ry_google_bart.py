import os
import sys
from transformers import AutoModelForAudio, AutoTokenizer

# Load the Whisper model
model = AutoModelForAudio.from_pretrained("openai/whisper-medium")
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-medium")

# Read the audio file
audio_file = "ry35words.wav"

# Transcribe the audio file
with open(audio_file, "rb") as f:
    audio = f.read()

# Decode the audio
text = model.transcribe(audio, tokenizer=tokenizer)

# Print the text
print(text)
