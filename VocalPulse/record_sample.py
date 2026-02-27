# record_sample.py
# Helper script to record a labeled training sample from your microphone.
#
# Usage:
#   python record_sample.py human  sample_01    (records to data/train/human/sample_01.wav)
#   python record_sample.py ai     my_ai_voice  (records to data/train/ai/my_ai_voice.wav)

import sys
import os
import numpy as np
import sounddevice as sd
import soundfile as sf

def main():
    if len(sys.argv) < 3:
        print("Usage: python record_sample.py <human|ai> <sample_name>")
        sys.exit(1)

    label = sys.argv[1].lower()
    name  = sys.argv[2]

    if label not in ("human", "ai"):
        print("Label must be 'human' or 'ai'")
        sys.exit(1)

    sr = 16000
    duration = 5  # seconds

    out_dir = os.path.join("data", "train", label)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.wav")

    print(f"Recording {duration}s of '{label}' voice → {out_path}")
    print("3... ", end="", flush=True)
    import time; time.sleep(1)
    print("2... ", end="", flush=True); time.sleep(1)
    print("1... ", end="", flush=True); time.sleep(1)
    print("GO!")

    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()

    audio = recording.flatten()
    sf.write(out_path, audio, sr)
    print(f"  ✓ Saved: {out_path}")
    print(f"\nWhen ready, run: python train.py")

if __name__ == "__main__":
    main()
