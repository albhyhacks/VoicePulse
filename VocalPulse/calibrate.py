import json
import librosa
import numpy as np
import sounddevice as sd
from src.core import VocalPulseCore

core = VocalPulseCore()
sr = 16000

# Record human
print("Speak now (4 seconds)...")
recording = sd.rec(int(4 * sr), samplerate=sr, channels=1, dtype=np.float32)
sd.wait()
human_audio = recording.flatten()

# Load AI
ai_audio, _ = librosa.load("data/ai_test.mp3", sr=sr)

# Analyze both
human_res = core.analyze(human_audio)
ai_res = core.analyze(ai_audio)

print("\n{:<25} {:>15} {:>15}".format("Feature", "HUMAN", "AI"))
print("-" * 55)
for key in human_res:
    h = human_res[key]
    a = ai_res[key]
    if isinstance(h, float):
        print("{:<25} {:>15.6f} {:>15.6f}".format(key, h, a))
    else:
        print("{:<25} {:>15} {:>15}".format(key, str(h), str(a)))
