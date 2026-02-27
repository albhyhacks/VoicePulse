# collect_and_train.py
# One-shot data collection + training session.
# Records N human voice samples, then trains the model.
#
# Usage:
#   python collect_and_train.py

import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from src.trainer import train

SR = 16000
DURATION = 4          # seconds per recording
NUM_SAMPLES = 8       # number of human voice recordings to collect
OUT_DIR = "data/train/human"

PROMPTS = [
    "Please say: 'Hello, my name is a real person speaking naturally.'",
    "Please say: 'The quick brown fox jumps over the lazy dog.'",
    "Please say: 'I am recording my voice for training purposes.'",
    "Please say: 'One two three four five six seven eight nine ten.'",
    "Please say: 'Can you hear me? This is my natural speaking voice.'",
    "Please say: 'The weather today is quite nice and sunny outside.'",
    "Please count slowly from one to ten in your normal voice.",
    "Please say: 'This voice detection system should recognize me as human.'",
    "Please say your full name and where you are from.",
    "Please say: 'Testing testing, one two three, this is a microphone test.'",
]


def record_one(index: int, prompt: str) -> np.ndarray:
    print(f"\n[{index}/{NUM_SAMPLES}] {prompt}")
    for c in ["3", "2", "1", "▶ SPEAK NOW"]:
        print(f"  {c}", flush=True)
        time.sleep(1)
    recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype=np.float32)
    sd.wait()
    return recording.flatten()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("data/train/ai", exist_ok=True)

    print("=" * 55)
    print("  VocalPulse — Data Collection")
    print(f"  Recording {NUM_SAMPLES} samples × {DURATION}s each")
    print("  Speak clearly and naturally when prompted.")
    print("=" * 55)

    saved = 0
    for i in range(NUM_SAMPLES):
        prompt = PROMPTS[i % len(PROMPTS)]
        audio = record_one(i + 1, prompt)

        # Quick energy check — warn if sample is silent
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            print("  ⚠  Very low energy detected — was the mic muted?")

        path = os.path.join(OUT_DIR, f"human_{i+1:03d}.wav")
        sf.write(path, audio, SR)
        print(f"  ✓ Saved → {path}  (RMS={rms:.4f})")
        saved += 1

    print(f"\n  Collected {saved} human samples.")

    # Check how many AI samples exist
    ai_dir = "data/train/ai"
    ai_count = sum(
        1 for f in os.listdir(ai_dir)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
    ) if os.path.isdir(ai_dir) else 0
    print(f"  AI samples in {ai_dir}: {ai_count}")

    if ai_count == 0:
        print("\n  ⚠  No AI samples found! Copy some AI voice files to data/train/ai/")
        print("     Then run:  python train.py")
        return

    print("\n" + "=" * 55)
    print("  Starting training...")
    print("=" * 55)
    train()


if __name__ == "__main__":
    main()
