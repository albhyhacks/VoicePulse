# src/features.py
# Shared feature-extraction utilities used by both the trainer and runtime.

import numpy as np
from src.core import VocalPulseCore

FEATURE_NAMES = [
    "f0_variation",
    "physiological_tremor",
    "breathing_cycle",
    "prespeech_activation",
    "jitter",
    "shimmer",
    "cognitive_timing",
    "spectral_flatness",
    "noise_floor",
    "hf_noise",
    "spectral_consistency",
]


def extract_feature_vector(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Run VocalPulseCore analysis on audio and return an ordered numpy feature
    vector aligned with FEATURE_NAMES.

    Parameters
    ----------
    audio : np.ndarray  mono float32/64 audio samples
    sr    : int         sample rate (should match VocalPulseCore default)

    Returns
    -------
    np.ndarray of shape (len(FEATURE_NAMES),)
    """
    core = VocalPulseCore(sr=sr)
    result = core.analyze(audio)
    return np.array([result[name] for name in FEATURE_NAMES], dtype=np.float32)
