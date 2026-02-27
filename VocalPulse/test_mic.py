import numpy as np
import sounddevice as sd
from src.classifier import VocalPulseClassifier

clf = VocalPulseClassifier()

sr = 16000
duration = 4

print("Speak now...")
recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
sd.wait()

audio = recording.flatten()
result = clf.predict(audio, sr=sr)

print(f"\nMethod:     {result['method']}")
print(f"Is Human:   {result['is_human']}")
print(f"Confidence: {result['confidence']:.1%}")
print("\nFeatures:")
for k, v in result["features"].items():
    print(f"  {k}: {v:.6f}")