import librosa
from src.classifier import VocalPulseClassifier

clf = VocalPulseClassifier()

audio, sr = librosa.load("data/ai_test.mp3", sr=16000)
result = clf.predict(audio, sr=sr)

print(f"\nMethod:     {result['method']}")
print(f"Is Human:   {result['is_human']}")
print(f"Confidence: {result['confidence']:.1%}")
print("\nFeatures:")
for k, v in result["features"].items():
    print(f"  {k}: {v:.6f}")