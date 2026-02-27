# import_and_train.py
# Import existing audio files into the training dataset and train the model.
#
# HOW TO USE:
#   1. Edit the HUMAN_FILES and AI_FILES lists below with absolute paths
#      to your audio files. Use forward slashes or raw strings.
#
#   2. Run:  python import_and_train.py
#
# Supported formats: .wav  .mp3  .flac  .ogg  .m4a
# ---------------------------------------------------------------------------

HUMAN_FILES = [
    # Add the absolute paths to YOUR VOICE recordings here, e.g.:
    # r"C:\Users\Abel\Downloads\my_voice_1.wav",
    # r"C:\Users\Abel\Downloads\my_voice_2.mp3",
]

AI_FILES = [
    # Add absolute paths to AI / TTS voice files here, e.g.:
    # r"C:\Users\Abel\Downloads\elevenlabs_output.mp3",
    # Leave empty to use files already in data/train/ai/
]

# ---------------------------------------------------------------------------
# You can also point to entire FOLDERS instead of individual files:
HUMAN_FOLDERS = [
    # r"C:\Users\Abel\Downloads\my_voice_recordings",
]

AI_FOLDERS = [
    # r"C:\Users\Abel\Downloads\ai_voices",
]
# ---------------------------------------------------------------------------

import os
import shutil
import glob
from src.trainer import train

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def collect_from_folders(folders):
    files = []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"  [WARNING] folder not found: {folder}")
            continue
        for f in os.listdir(folder):
            if f.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(folder, f))
    return files


def import_files(file_list, dest_dir, label):
    os.makedirs(dest_dir, exist_ok=True)
    imported = 0
    for src in file_list:
        if not os.path.isfile(src):
            print(f"  [WARNING] file not found: {src}")
            continue
        fname = os.path.basename(src)
        dst   = os.path.join(dest_dir, fname)
        shutil.copy2(src, dst)
        print(f"  ✓ [{label}] {fname}")
        imported += 1
    return imported


def main():
    print("=" * 55)
    print("  VocalPulse — Import & Train")
    print("=" * 55)

    human_files = list(HUMAN_FILES) + collect_from_folders(HUMAN_FOLDERS)
    ai_files    = list(AI_FILES)    + collect_from_folders(AI_FOLDERS)

    print(f"\nImporting {len(human_files)} human file(s)...")
    n_h = import_files(human_files, "data/train/human", "human")

    print(f"Importing {len(ai_files)} AI file(s)...")
    n_a = import_files(ai_files, "data/train/ai", "ai")

    # Count total after import
    existing_human = [
        f for f in glob.glob("data/train/human/**/*", recursive=True)
        if f.lower().endswith(AUDIO_EXTS)
    ]
    existing_ai = [
        f for f in glob.glob("data/train/ai/**/*", recursive=True)
        if f.lower().endswith(AUDIO_EXTS)
    ]

    print(f"\nTraining set: {len(existing_human)} human, {len(existing_ai)} AI files")

    if len(existing_human) == 0:
        print("\n[ERROR] No human files found.")
        print("  Edit HUMAN_FILES or HUMAN_FOLDERS at the top of this script")
        print("  OR manually copy your voice recordings to: data/train/human/")
        return

    if len(existing_ai) == 0:
        print("\n[ERROR] No AI files found.")
        print("  Edit AI_FILES or AI_FOLDERS at the top of this script")
        print("  OR manually copy AI voice files to: data/train/ai/")
        return

    print("\n" + "=" * 55)
    train()


if __name__ == "__main__":
    main()
