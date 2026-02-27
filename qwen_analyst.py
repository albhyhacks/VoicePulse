# backend/Qwen/qwen_analyst.py
# Featherless AI client — Qwen 2.5-7B-Instruct analyst layer
# Role: explain WHY a voice was flagged, generate challenges, report fraud intelligence

import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from parent backend/ directory
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_env_path)


class QwenVoiceAnalyst:
    """
    Intelligent analyst layer using Featherless AI (Qwen 2.5-7B-Instruct).
    Does NOT detect audio — it explains and advises based on signal results.
    """

    def __init__(self):
        api_key = os.getenv("FEATHERLESS_API_KEY")
        base_url = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
        self.model = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")

        if not api_key:
            raise ValueError("FEATHERLESS_API_KEY not set in .env file")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _call(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        """Low-level API call to Featherless."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[AI analyst unavailable: {str(e)}]"

    # ------------------------------------------------------------------ #
    #  1. EXPLAIN ANALYSIS — plain-language breakdown of biomarker results  #
    # ------------------------------------------------------------------ #
    def explain_analysis(self, result: dict, language: str = "english") -> dict:
        """
        Given biomarker detection results, produce a human-readable explanation
        of WHY the verdict was reached. Output in the requested language.
        """
        lang_map = {
            "tamil":     "Tamil (தமிழ்)",
            "malayalam": "Malayalam (മലയാളം)",
            "hindi":     "Hindi (हिन्दी)",
            "english":   "English"
        }
        output_lang = lang_map.get(language.lower(), "English")

        verdict        = result.get("verdict", "UNKNOWN")
        score          = result.get("authenticity_score", 0.0)
        biomarkers     = result.get("biomarkers", {})
        tremor         = biomarkers.get("tremor", {})
        respiratory    = biomarkers.get("respiratory", {})
        precursor      = biomarkers.get("precursor", {})
        jitter         = biomarkers.get("jitter", {})

        system_prompt = (
            "You are VocalPulse's AI Analyst. Your role is to explain voice biometric "
            "analysis results in clear, accessible language. You are NOT the detector — "
            "a physics-based signal processor already made the verdict. You explain WHY. "
            "Be precise, factual, and reassuring. Output ONLY in the requested language. "
            "Keep explanations under 3 sentences."
        )

        user_prompt = f"""
Explain this voice authentication result to the user in {output_lang}:

VERDICT: {verdict}
CONFIDENCE: {score:.1%}

BIOMARKER SIGNALS:
- Physiological Tremor (8-12 Hz): detected={tremor.get('is_human', False)}, score={tremor.get('score', 0):.2f}
- Breathing Rhythm (0.1-0.5 Hz): detected={respiratory.get('is_human', False)}, depth={respiratory.get('depth', 0):.3f}
- Neural Precursor (50-200ms): detected={precursor.get('is_live_human', False)}, delay={precursor.get('delay_ms', 0):.0f}ms
- Vocal Jitter: detected={jitter.get('is_human', False)}, value={jitter.get('value', 0):.4f}

Give a 2-3 sentence explanation of why this verdict was reached. Focus on what the signals mean, not the numbers.
Also provide a one-sentence structured JSON at the end with keys: verdict_summary, key_evidence, confidence_level (high/medium/low).
Format the JSON as: JSON: {{"verdict_summary": "...", "key_evidence": "...", "confidence_level": "..."}}
"""

        raw = self._call(system_prompt, user_prompt, max_tokens=400)

        # Parse out structured JSON if present
        explanation = raw
        structured  = {}
        json_match  = re.search(r'JSON:\s*(\{.*?\})', raw, re.DOTALL)
        if json_match:
            try:
                structured  = json.loads(json_match.group(1))
                explanation = raw[:json_match.start()].strip()
            except json.JSONDecodeError:
                pass

        return {
            "explanation": explanation,
            "structured":  structured,
            "language":    language
        }

    # ------------------------------------------------------------------ #
    #  2. FRAUD INTELLIGENCE — deep report when synthetic is detected      #
    # ------------------------------------------------------------------ #
    def analyze_fraud_pattern(self, result: dict) -> dict:
        """
        When a voice is flagged SYNTHETIC, generate intelligence about the
        likely attack type, severity, and recommended action.
        """
        biomarkers  = result.get("biomarkers", {})
        tremor      = biomarkers.get("tremor", {})
        respiratory = biomarkers.get("respiratory", {})
        precursor   = biomarkers.get("precursor", {})
        jitter      = biomarkers.get("jitter", {})
        score       = result.get("authenticity_score", 0.0)

        system_prompt = (
            "You are a voice fraud intelligence analyst. When AI-generated or cloned "
            "audio is detected, you identify the attack pattern and recommend action. "
            "Be concise and security-focused."
        )

        user_prompt = f"""
A synthetic voice was detected. Analyze the fraud pattern:

CONFIDENCE IT'S SYNTHETIC: {(1 - score):.1%}
MISSING BIOLOGICAL SIGNALS:
- Tremor absent: {not tremor.get('is_human', True)}
- Breathing absent: {not respiratory.get('is_human', True)}
- Neural precursor absent: {not precursor.get('is_live_human', True)}
- Natural jitter absent: {not jitter.get('is_human', True)}
Missing signals count: {sum([not tremor.get('is_human', True), not respiratory.get('is_human', True), not precursor.get('is_live_human', True), not jitter.get('is_human', True)])} out of 4

Provide:
1. Likely attack type (TTS synthesis / voice cloning / replay attack / deepfake)
2. Severity (low/medium/high/critical)
3. Recommended action for the security team (one sentence)

Format as: ATTACK: ...\nSEVERITY: ...\nACTION: ...\nANALYSIS: (2 sentence explanation)
"""

        raw = self._call(system_prompt, user_prompt, max_tokens=300)

        # Parse structured fields
        attack   = self._extract_field(raw, "ATTACK")
        severity = self._extract_field(raw, "SEVERITY")
        action   = self._extract_field(raw, "ACTION")
        analysis = self._extract_field(raw, "ANALYSIS")

        return {
            "fraud_analysis":       analysis or raw,
            "attack_type":          attack  or "Unknown synthetic",
            "severity":             severity or "high",
            "recommended_action":   action  or "Block request and trigger manual review",
            "raw_response":         raw
        }

    # ------------------------------------------------------------------ #
    #  3. CHALLENGE GENERATION — AI-generated cognitive challenge          #
    # ------------------------------------------------------------------ #
    def generate_challenge(self, difficulty: str = "medium",
                           language: str = "english",
                           previous_challenges: list = None) -> dict:
        """
        Generate a unique cognitive-motor challenge that a human can respond to
        with the expected natural delay (150-400ms for cognitive-motor processing).
        """
        previous_challenges = previous_challenges or []
        lang_map = {
            "tamil":     "Tamil",
            "malayalam": "Malayalam",
            "hindi":     "Hindi",
            "english":   "English"
        }
        output_lang = lang_map.get(language.lower(), "English")

        difficulty_config = {
            "easy":   {"delay": "300-700ms",  "load": "low",    "desc": "simple recall"},
            "medium": {"delay": "500-1200ms", "load": "medium", "desc": "mild cognitive processing"},
            "hard":   {"delay": "700-1800ms", "load": "high",   "desc": "heavy cognitive load (Stroop, math)"}
        }
        cfg = difficulty_config.get(difficulty, difficulty_config["medium"])

        system_prompt = (
            "You are designing cognitive-motor challenge prompts for a voice biometric system. "
            "Challenges must require human cognitive processing before speaking — they exploit "
            "natural neural delay (150-400ms) that AI synthesizers cannot fake. "
            "Generate unique, unpredictable challenges."
        )

        prev_str = "\n".join(f"- {c}" for c in previous_challenges) if previous_challenges else "None"

        user_prompt = f"""
Generate a voice authentication challenge:
- Difficulty: {difficulty} ({cfg['desc']})
- Expected human response delay: {cfg['delay']}
- Language for the challenge text: {output_lang}
- Cognitive load: {cfg['load']}
- Previous challenges to AVOID repeating: 
{prev_str}

The challenge should require a human to think before speaking (not just read text aloud).
Examples: color-word conflicts (Stroop), simple math, name something in a category, finish a sentence unexpectedly.

Respond with:
CHALLENGE: (the prompt text in {output_lang})
DELAY_MS: (expected min-max as "min,max" in ms)
COGNITIVE_LOAD: (low/medium/high)
WHY_EFFECTIVE: (one sentence in English)
"""

        raw = self._call(system_prompt, user_prompt, max_tokens=250)

        challenge_text = self._extract_field(raw, "CHALLENGE") or "Say the color of the sky"
        delay_str      = self._extract_field(raw, "DELAY_MS")  or "500,1200"
        cog_load       = self._extract_field(raw, "COGNITIVE_LOAD") or difficulty
        why            = self._extract_field(raw, "WHY_EFFECTIVE") or ""

        try:
            delay_parts         = [int(x.strip()) for x in delay_str.split(",")]
            expected_delay_ms   = delay_parts if len(delay_parts) == 2 else [500, 1200]
        except ValueError:
            expected_delay_ms   = [500, 1200]

        return {
            "challenge_text":    challenge_text,
            "expected_delay_ms": expected_delay_ms,
            "cognitive_load":    cog_load,
            "why_effective":     why,
            "difficulty":        difficulty,
            "language":          language
        }

    # ------------------------------------------------------------------ #
    #  Helper                                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_field(text: str, field: str) -> str:
        """Extract a value from 'FIELD: value' formatted text."""
        pattern = rf'{field}:\s*(.+?)(?:\n|$)'
        match   = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
