# train.py
# Run this script to train the VocalPulse ML classifier.
#
# Usage:
#   python train.py
#
# Put your labeled audio files in:
#   data/train/human/   <- real human voice recordings (.wav, .mp3, ...)
#   data/train/ai/      <- AI-generated / TTS voice files
#
# The trained model will be saved to:
#   models/classifier.pkl

import sys
from src.trainer import train

if __name__ == "__main__":
    pipeline = train()
    if pipeline is None:
        sys.exit(1)
    print("\n  Done! Use python test_ai.py or python test_mic.py to test.\n")
