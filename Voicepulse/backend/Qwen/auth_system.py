# backend/Qwen/auth_system.py
# Core biomarker signal processor + Qwen AI explanation overlay
# Physics detects → Qwen explains

import numpy as np
from typing import Dict, List, Optional

try:
    from qwen_analyst import QwenVoiceAnalyst
except ImportError:
    # Allow import from different working directories
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from qwen_analyst import QwenVoiceAnalyst


# ────────────────────────────────────────────────────────────────────────────
#  SIGNAL PROCESSING ENGINE
#  Pure physics — no AI involved in the detection step
# ────────────────────────────────────────────────────────────────────────────

class BiomarkerAnalyzer:
    """
    Detects the 5 physiological biomarkers that only a living human body produces.
    AI voice cloners (ElevenLabs, XTTS, RVC) cannot replicate these because they
    have no muscles, lungs, or neurons.
    """

    TREMOR_LOW_HZ    = 8.0
    TREMOR_HIGH_HZ   = 12.0
    BREATH_LOW_HZ    = 0.1
    BREATH_HIGH_HZ   = 0.5
    PRECURSOR_MIN_MS = 50
    PRECURSOR_MAX_MS = 200

    def analyze(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """
        Run all 5 biomarker checks. Returns full result dict including verdict.
        """
        tremor      = self._check_tremor(audio, sr)
        respiratory = self._check_respiratory(audio, sr)
        precursor   = self._check_precursor(audio, sr)
        jitter      = self._check_jitter(audio, sr)
        shimmer     = self._check_shimmer(audio)

        signals_detected = sum([
            tremor['is_human'],
            respiratory['is_human'],
            precursor['is_live_human'],
            jitter['is_human'],
            shimmer['is_human']
        ])

        # Weighted authenticity score
        authenticity_score = (
            0.30 * tremor['score'] +
            0.20 * respiratory['score'] +
            0.20 * precursor['score'] +
            0.15 * jitter['score'] +
            0.15 * shimmer['score']
        )

        is_authentic = authenticity_score >= 0.60

        return {
            "verdict":            "HUMAN" if is_authentic else "SYNTHETIC",
            "is_authentic":       is_authentic,
            "authenticity_score": authenticity_score,
            "signals_detected":   signals_detected,
            "signals_total":      5,
            "biomarkers": {
                "tremor":      tremor,
                "respiratory": respiratory,
                "precursor":   precursor,
                "jitter":      jitter,
                "shimmer":     shimmer
            }
        }

    def _check_tremor(self, audio: np.ndarray, sr: int) -> Dict:
        """8-12 Hz physiological tremor from motor neuron firing in vocal cords."""
        try:
            fft  = np.abs(np.fft.rfft(audio))
            freq = np.fft.rfftfreq(len(audio), 1 / sr)

            band_mask  = (freq >= self.TREMOR_LOW_HZ) & (freq <= self.TREMOR_HIGH_HZ)
            total_mask = (freq >= 1) & (freq <= 100)

            band_power  = np.sum(fft[band_mask]  ** 2)
            total_power = np.sum(fft[total_mask] ** 2)

            ratio    = float(band_power / total_power) if total_power > 0 else 0.0
            score    = min(ratio * 50, 1.0)          # Normalise empirically
            is_human = score > 0.25

            return {
                "is_human": is_human,
                "score":    score,
                "ratio":    ratio,
                "hz_peak":  float(freq[band_mask][np.argmax(fft[band_mask])]) if band_mask.any() else 0.0
            }
        except Exception:
            return {"is_human": False, "score": 0.0, "ratio": 0.0, "hz_peak": 0.0}

    def _check_respiratory(self, audio: np.ndarray, sr: int) -> Dict:
        """0.1-0.5 Hz breathing rhythm — amplitude modulation from lungs."""
        try:
            envelope    = np.abs(audio)
            # Downsample envelope for low-freq analysis
            decimation  = max(1, sr // 100)
            env_down    = envelope[::decimation]
            sr_down     = sr / decimation

            fft  = np.abs(np.fft.rfft(env_down))
            freq = np.fft.rfftfreq(len(env_down), 1 / sr_down)

            band_mask  = (freq >= self.BREATH_LOW_HZ) & (freq <= self.BREATH_HIGH_HZ)
            total_mask = freq > 0

            band_power  = np.sum(fft[band_mask]  ** 2)
            total_power = np.sum(fft[total_mask] ** 2)

            depth    = float(band_power / total_power) if total_power > 0 else 0.0
            score    = min(depth * 5, 1.0)
            is_human = score > 0.20

            return {
                "is_human": is_human,
                "score":    score,
                "depth":    depth,
                "rhythm_hz": float(freq[band_mask][np.argmax(fft[band_mask])]) if band_mask.any() else 0.0
            }
        except Exception:
            return {"is_human": False, "score": 0.0, "depth": 0.0, "rhythm_hz": 0.0}

    def _check_precursor(self, audio: np.ndarray, sr: int) -> Dict:
        """
        50-200ms neural precursor — the muscle pre-activation burst before speech.
        AI generates audio from the first sample; humans need motor spin-up time.
        """
        try:
            window_ms   = 200
            window_samp = int(sr * window_ms / 1000)
            if len(audio) < window_samp * 2:
                return {"is_live_human": False, "score": 0.5, "delay_ms": 0.0}

            onset_energy  = np.mean(audio[:window_samp]  ** 2)
            speech_energy = np.mean(audio[window_samp:window_samp * 4] ** 2)

            # Humans: low energy → higher energy ramp; AI: flat or instant onset
            ratio    = float(onset_energy / speech_energy) if speech_energy > 0 else 0.0
            # A natural precursor has ratio ~0.05-0.4
            is_human = 0.02 <= ratio <= 0.50
            score    = 1.0 - abs(ratio - 0.15) / 0.5 if is_human else max(0, 0.5 - ratio)
            score    = float(np.clip(score, 0, 1))

            # Estimate delay
            energy   = np.array([np.mean(audio[i:i+100]**2) for i in range(0, window_samp*2, 100)])
            threshold = np.max(energy) * 0.1
            delay_idx = np.argmax(energy > threshold)
            delay_ms  = float(delay_idx * 100 / sr * 1000)

            return {
                "is_live_human": is_human,
                "score":         score,
                "delay_ms":      delay_ms,
                "onset_ratio":   ratio
            }
        except Exception:
            return {"is_live_human": False, "score": 0.0, "delay_ms": 0.0, "onset_ratio": 0.0}

    def _check_jitter(self, audio: np.ndarray, sr: int) -> Dict:
        """Natural biological imperfection in pitch — jitter (cycle-to-cycle F0 variation)."""
        try:
            # Zero-crossing rate as F0 proxy
            zcr = np.where(np.diff(np.sign(audio)))[0]
            if len(zcr) < 10:
                return {"is_human": False, "score": 0.0, "value": 0.0}

            periods = np.diff(zcr).astype(float)
            jitter  = float(np.std(periods) / (np.mean(periods) + 1e-8))

            # Human jitter: 0.5%-4%; AI typically near 0 (too perfect)
            is_human = 0.005 <= jitter <= 0.08
            if jitter < 0.005:
                score = jitter / 0.005 * 0.5
            elif jitter <= 0.04:
                score = 1.0
            else:
                score = max(0, 1.0 - (jitter - 0.04) / 0.04)

            return {
                "is_human": is_human,
                "score":    float(np.clip(score, 0, 1)),
                "value":    jitter
            }
        except Exception:
            return {"is_human": False, "score": 0.0, "value": 0.0}

    def _check_shimmer(self, audio: np.ndarray) -> Dict:
        """Natural biological imperfection in amplitude — shimmer (loudness variation)."""
        try:
            frame_size = 512
            rms_values = [
                np.sqrt(np.mean(audio[i:i+frame_size]**2))
                for i in range(0, len(audio) - frame_size, frame_size // 2)
            ]
            if len(rms_values) < 4:
                return {"is_human": False, "score": 0.0, "value": 0.0}

            rms = np.array(rms_values)
            shimmer = float(np.std(rms) / (np.mean(rms) + 1e-8))

            # Human shimmer: 2%-10%; AI: too stable or artificially varying
            is_human = 0.02 <= shimmer <= 0.20
            if shimmer < 0.02:
                score = shimmer / 0.02 * 0.5
            elif shimmer <= 0.10:
                score = 1.0
            else:
                score = max(0, 1.0 - (shimmer - 0.10) / 0.10)

            return {
                "is_human": is_human,
                "score":    float(np.clip(score, 0, 1)),
                "value":    shimmer
            }
        except Exception:
            return {"is_human": False, "score": 0.0, "value": 0.0}


# ────────────────────────────────────────────────────────────────────────────
#  VOCALPULSE AUTH — Signal Processing + Qwen AI Explanation
# ────────────────────────────────────────────────────────────────────────────

class VocalPulseAuth:
    """
    Two-layer authentication:
    Layer 1 — Physics: BiomarkerAnalyzer detects biological signals
    Layer 2 — AI:     QwenVoiceAnalyst explains the verdict in natural language
    """

    def __init__(self, use_ai_analyst: bool = True, language: str = "english"):
        self.analyzer          = BiomarkerAnalyzer()
        self.language          = language
        self.challenge_history = []

        self.qwen = None
        if use_ai_analyst:
            try:
                self.qwen = QwenVoiceAnalyst()
            except Exception as e:
                print(f"[VocalPulse] Qwen analyst unavailable: {e}. Running without AI overlay.")

    def authenticate_full(self, user_id: str, audio: np.ndarray,
                          sr: int = 16000) -> Dict:
        """
        Full authentication pipeline:
        1) Biomarker signal processing (physics — always runs)
        2) Qwen AI explanation (if available)
        3) Fraud intelligence if SYNTHETIC
        """
        result = self.analyzer.analyze(audio, sr)
        result["user_id"] = user_id

        if self.qwen:
            try:
                ai_insight = self.qwen.explain_analysis(result, self.language)
                result["ai_explanation"] = ai_insight["explanation"]
                result["ai_structured"]  = ai_insight["structured"]
                result["ai_language"]    = ai_insight["language"]
            except Exception as e:
                result["ai_explanation"] = f"AI explanation unavailable: {e}"

            if not result["is_authentic"]:
                try:
                    fraud = self.qwen.analyze_fraud_pattern(result)
                    result["fraud_intelligence"] = fraud
                except Exception as e:
                    result["fraud_intelligence"] = {"error": str(e)}

        return result

    def get_smart_challenge(self, difficulty: str = "medium") -> Dict:
        """Return an AI-generated (or fallback) cognitive challenge."""
        if not self.qwen:
            return self._static_challenge(difficulty)

        try:
            challenge = self.qwen.generate_challenge(
                difficulty=difficulty,
                language=self.language,
                previous_challenges=self.challenge_history[-5:]
            )
            self.challenge_history.append(challenge.get("challenge_text", ""))
            return challenge
        except Exception:
            return self._static_challenge(difficulty)

    def authenticate_with_challenge(self, user_id: str, audio: np.ndarray,
                                    response_time_ms: float,
                                    challenge_text: str) -> Dict:
        """Full challenge-response: biomarkers + timing analysis."""
        result          = self.authenticate_full(user_id, audio)
        expected_delay  = self._get_expected_delay(challenge_text)
        timing          = self._analyze_timing(response_time_ms, expected_delay)
        result["timing_analysis"] = timing

        final_score = 0.7 * result["authenticity_score"] + 0.3 * timing["human_score"]
        result["final_score"]   = final_score
        result["final_verdict"] = "HUMAN" if final_score > 0.65 else "SYNTHETIC"
        return result

    # ── Internal helpers ─────────────────────────────────────────────────── #

    @staticmethod
    def _analyze_timing(actual_ms: float, expected_range: List[int]) -> Dict:
        min_d, max_d = expected_range
        in_range     = min_d <= actual_ms <= max_d

        if actual_ms < 100:
            human_score = 0.10   # Impossibly fast — replay/AI
        elif actual_ms < min_d:
            human_score = 0.35
        elif in_range:
            human_score = 0.92
        elif actual_ms > max_d * 2:
            human_score = 0.40   # Too slow — possibly recorded
        else:
            human_score = 0.65

        return {
            "response_time_ms": actual_ms,
            "expected_range":   expected_range,
            "in_range":         in_range,
            "human_score":      human_score,
            "assessment":       "normal" if in_range else "anomalous"
        }

    @staticmethod
    def _static_challenge(difficulty: str) -> Dict:
        challenges = {
            "easy": {
                "challenge_text":    "Say your full name slowly",
                "expected_delay_ms": [300, 800],
                "cognitive_load":    "low",
                "why_effective":     "Name recall triggers autobiographical memory access."
            },
            "medium": {
                "challenge_text":    "What is 17 plus 28?",
                "expected_delay_ms": [500, 1200],
                "cognitive_load":    "medium",
                "why_effective":     "Mental arithmetic forces genuine cognitive processing."
            },
            "hard": {
                "challenge_text":    "Say the word RED but think of the color BLUE",
                "expected_delay_ms": [700, 1800],
                "cognitive_load":    "high",
                "why_effective":     "Stroop-effect conflict requires suppression of dominant response."
            }
        }
        return challenges.get(difficulty, challenges["medium"])

    @staticmethod
    def _get_expected_delay(challenge_text: str) -> List[int]:
        text = challenge_text.lower()
        if any(w in text for w in ["color", "colour", "blue", "red", "green"]):
            return [700, 1800]
        if any(w in text for w in ["plus", "minus", "times", "divide", "multiply"]):
            return [500, 1200]
        return [300, 1000]
