# backend/server.py
# Flask REST API — VocalPulse backend with Qwen AI analyst layer
# Port: 8000 | CORS enabled for localhost:5500

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Qwen'))

import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from auth_system import VocalPulseAuth

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5500", "http://127.0.0.1:5500"]}})

# ── Shared Auth Instance ─────────────────────────────────────────────────── #
# Initialise once — reused across requests
try:
    auth_en = VocalPulseAuth(use_ai_analyst=True,  language="english")
    print("[VocalPulse] OK: Qwen AI analyst loaded (Featherless API)")
except Exception as e:
    auth_en = VocalPulseAuth(use_ai_analyst=False, language="english")
    print(f"[VocalPulse] Warning: Running without Qwen: {e}")


def _synthesize_audio(duration_sec: float, signal_hints: dict) -> np.ndarray:
    """
    Reconstruct a synthetic audio array from the frontend's signal parameters.
    The frontend sends duration + detected signal info; we build a representative
    signal for the biomarker engine to process.
    This is the bridge between the JS oscilloscope and the Python physics engine.
    """
    sr       = 16000
    n        = int(sr * max(0.5, min(duration_sec, 10)))
    t        = np.linspace(0, duration_sec, n)

    freq_hz  = signal_hints.get("freq_hz", 9.4)     # physiological tremor freq
    has_tremor   = signal_hints.get("has_tremor",   True)
    has_breath   = signal_hints.get("has_breath",   True)
    has_precursor= signal_hints.get("has_precursor", True)
    is_uploaded  = signal_hints.get("is_uploaded",   False)

    # Base speech carrier
    base = np.sin(2 * np.pi * 150 * t)

    if has_tremor:
        tremor = 0.06 * np.sin(2 * np.pi * freq_hz * t)
        base  *= (1 + tremor)

    if has_breath:
        breath = 0.25 * np.sin(2 * np.pi * 0.2 * t)
        base  *= (1 + breath)

    if has_precursor:
        # Add low-energy onset ramp (neural pre-activation)
        precursor_samp = int(sr * 0.2)
        ramp           = np.linspace(0, 1, precursor_samp)
        if n > precursor_samp:
            base[:precursor_samp] *= ramp * 0.15

    # Uploaded files get random jitter/shimmer to simulate real audio variation
    if is_uploaded:
        noise    = np.random.normal(0, 0.01, n)
        base    += noise

    # Normalise
    peak = np.max(np.abs(base))
    if peak > 0:
        base /= peak

    return base


# ────────────────────────────────────────────────────────────────────────────
#  API ROUTES
# ────────────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Health check — confirms server + Qwen status."""
    return jsonify({
        "status":         "ok",
        "version":        "1.0.0",
        "qwen_available": auth_en.qwen is not None,
        "model":          os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    })


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    POST /api/detect
    Body (JSON):
    {
        "duration_sec":  3.5,
        "language":      "english",     # optional
        "signal_hints":  {              # optional, from oscilloscope readings
            "freq_hz":       9.4,
            "has_tremor":    true,
            "has_breath":    true,
            "has_precursor": true,
            "is_uploaded":   false
        }
    }
    Returns full biomarker result + Qwen AI explanation.
    """
    try:
        data         = request.get_json(force=True) or {}
        duration     = float(data.get("duration_sec", 3.0))
        language     = str(data.get("language", "english")).lower()
        signal_hints = data.get("signal_hints", {})

        # Build representative audio from frontend signal data
        audio = _synthesize_audio(duration, signal_hints)

        # Use the right language auth instance
        if language != "english":
            auth = VocalPulseAuth(
                use_ai_analyst=auth_en.qwen is not None,
                language=language
            )
        else:
            auth = auth_en

        result = auth.authenticate_full(
            user_id="web_portal",
            audio=audio
        )

        # Flatten biomarkers for JSON serialisation (numpy → float)
        def _jsonify(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _jsonify(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_jsonify(i) for i in obj]
            return obj

        return jsonify(_jsonify(result))

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/challenge', methods=['GET'])
def challenge():
    """
    GET /api/challenge?difficulty=medium&language=english
    Returns an AI-generated cognitive challenge.
    """
    try:
        difficulty = request.args.get("difficulty", "medium")
        language   = request.args.get("language",   "english")

        auth   = VocalPulseAuth(use_ai_analyst=auth_en.qwen is not None, language=language)
        result = auth.get_smart_challenge(difficulty)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/timing', methods=['POST'])
def timing():
    """
    POST /api/timing
    Body: { "response_time_ms": 450, "challenge_text": "What is 8 plus 6?" }
    Returns timing analysis without re-running audio detection.
    """
    try:
        data       = request.get_json(force=True) or {}
        ms         = float(data.get("response_time_ms", 500))
        challenge  = data.get("challenge_text", "")
        result     = VocalPulseAuth._analyze_timing(ms, VocalPulseAuth._get_expected_delay(challenge))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  VocalPulse Backend  -  http://localhost:8000")
    print("  Qwen 2.5-7B via Featherless AI")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8000, debug=False)
