import transformers
import librosa
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("Loading model...")
pipe = transformers.pipeline(
    model='fixie-ai/ultravox-v0_5-llama-3_2-1b', 
    trust_remote_code=True,
    device_map="auto" # use GPU if available
)

print("Creating dummy audio...")
# 1 second of silence
sr = 16000
audio = np.zeros(sr, dtype=np.float32)

turns = [
    {
        "role": "system",
        "content": "You are a highly accurate transcription assistant. Transcribe the following audio exactly as spoken."
    },
    {
        "role": "user",
        "content": "Transcribe this audio: <|audio|>"
    }
]

print("Running pipeline...")
result = pipe(
    {'audio': audio, 'turns': turns, 'sampling_rate': sr}, 
    max_new_tokens=30
)

print("Result format:")
print(result)
