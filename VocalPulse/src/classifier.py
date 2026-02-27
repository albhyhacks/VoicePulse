# src/classifier.py
# Runtime classifier: loads the trained model and wraps it with a clean API.
# Falls back to rule-based scoring if no model is found.

import os
import numpy as np

MODEL_PATH = "models/classifier.pkl"


class VocalPulseClassifier:
    """
    Wraps a trained scikit-learn pipeline (scaler + RandomForest).
    Falls back to VocalPulseCore rule-based scoring if no model is available.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                import joblib
                self.pipeline = joblib.load(self.model_path)
                print(f"[VocalPulse] ML model loaded from {self.model_path}")
            except Exception as e:
                print(f"[VocalPulse] Could not load model: {e} — using rule-based fallback")
        else:
            print(f"[VocalPulse] No trained model at {self.model_path} — using rule-based scoring")

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def predict(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """
        Analyze audio and return classification result.

        Returns
        -------
        dict with keys:
            is_human     : bool
            confidence   : float  (0.0–1.0, probability of predicted class)
            method       : "ml" | "rule_based"
            features     : dict of raw feature values
        """
        from src.features import extract_feature_vector, FEATURE_NAMES
        from src.core import VocalPulseCore

        core = VocalPulseCore(sr=sr)
        result = core.analyze(audio)

        features = {k: result[k] for k in FEATURE_NAMES}

        if self.is_trained:
            vec = np.array([features[k] for k in FEATURE_NAMES]).reshape(1, -1)
            label = int(self.pipeline.predict(vec)[0])
            proba = self.pipeline.predict_proba(vec)[0]
            # proba[0] = P(AI), proba[1] = P(human)
            is_human = bool(label == 1)
            confidence = float(proba[1] if is_human else proba[0])
            method = "ml"
        else:
            is_human = bool(result["is_human"])
            score = result.get("liveness_score", 0.0)
            # Rough confidence from rule-based score distance from 0
            confidence = float(min(abs(score) / 60.0, 1.0))
            method = "rule_based"

        return {
            "is_human": is_human,
            "confidence": confidence,
            "method": method,
            "features": features,
            # Include raw scoring fields if coming from rule-based
            **({
                "human_evidence": result.get("human_evidence"),
                "ai_evidence": result.get("ai_evidence"),
                "liveness_score": result.get("liveness_score"),
            } if method == "rule_based" else {}),
        }
