# src/trainer.py
# Train a RandomForest classifier to distinguish human vs AI voices.
# Scans data/train/human/ and data/train/ai/ for audio files,
# extracts features, trains, evaluates, and saves the model.

import os
import glob
import numpy as np
import joblib
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features import extract_feature_vector, FEATURE_NAMES

AUDIO_EXTS = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
MODEL_PATH = "models/classifier.pkl"
DATA_ROOT = "data/train"


def collect_files(folder: str) -> list[str]:
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return files


def load_dataset(data_root: str = DATA_ROOT, sr: int = 16000):
    """
    Load all labeled audio from data_root/human/ and data_root/ai/.
    Returns X (n_samples, n_features) and y (n_samples,) with 1=human, 0=AI.
    """
    X, y, paths = [], [], []

    for label, folder_name in [(1, "human"), (0, "ai")]:
        folder = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder):
            print(f"  [WARNING] folder not found: {folder}")
            continue

        files = collect_files(folder)
        if not files:
            print(f"  [WARNING] no audio files found in: {folder}")
            continue

        print(f"\n  Loading {len(files)} '{folder_name}' files...")
        for path in files:
            try:
                audio, _ = librosa.load(path, sr=sr, mono=True)
                vec = extract_feature_vector(audio, sr=sr)
                X.append(vec)
                y.append(label)
                paths.append(path)
                print(f"    ✓ {os.path.basename(path)}")
            except Exception as e:
                print(f"    ✗ {os.path.basename(path)}: {e}")

    return np.array(X), np.array(y), paths


def train(data_root: str = DATA_ROOT, model_path: str = MODEL_PATH):
    print("=" * 55)
    print("  VocalPulse — Training classifier")
    print("=" * 55)

    X, y, paths = load_dataset(data_root)

    n_human = int(np.sum(y == 1))
    n_ai    = int(np.sum(y == 0))
    print(f"\n  Dataset: {n_human} human, {n_ai} AI samples")

    if n_human == 0 or n_ai == 0:
        print("\n  [ERROR] Need samples from BOTH classes to train.")
        print(f"  Put human voice files in:  {data_root}/human/")
        print(f"  Put AI voice files in:     {data_root}/ai/")
        return None

    # Build pipeline: optional scaler + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",   # handles unequal class sizes
            random_state=42,
        ))
    ])

    # Cross-validation (use leave-one-out if dataset is very small)
    n_splits = min(5, n_human, n_ai)
    if n_splits < 2:
        print("\n  [WARNING] Only 1 sample per class — skipping cross-validation.")
        pipeline.fit(X, y)
        cv_mean, cv_std = float("nan"), float("nan")
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")
        cv_mean, cv_std = scores.mean(), scores.std()
        pipeline.fit(X, y)

    print(f"\n  Cross-val F1: {cv_mean:.3f} ± {cv_std:.3f}")

    # Full training report
    y_pred = pipeline.predict(X)
    print("\n  Training report (in-sample):")
    print(classification_report(y, y_pred, target_names=["AI", "Human"]))

    # Feature importances
    rf = pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    print("  Feature importances:")
    for i in order:
        bar = "█" * int(importances[i] * 40)
        print(f"    {FEATURE_NAMES[i]:25s} {importances[i]:.3f}  {bar}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\n  ✓ Model saved → {model_path}")
    return pipeline
