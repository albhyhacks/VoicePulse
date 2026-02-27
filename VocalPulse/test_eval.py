import sys
import json
import librosa
from src.core import VocalPulseCore

def main():
    core = VocalPulseCore()
    
    try:
        audio, sr = librosa.load("data/ai_test.mp3", sr=16000)
        res_ai = core.analyze(audio)
    except Exception as e:
        res_ai = {"error": str(e)}

    with open("results.json", "w") as f:
        json.dump(res_ai, f, indent=4)
        
if __name__ == "__main__":
    main()
