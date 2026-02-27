# src/core.py

import numpy as np
import librosa
from scipy import signal


class VocalPulseCore:
    def __init__(self, sr=16000):
        self.sr = sr

    # -----------------------------------------------------------
    # Step 1: Extract pitch contour (F0)
    # -----------------------------------------------------------
    def extract_f0(self, audio):
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=80,
            fmax=600,
            sr=self.sr,
            hop_length=160
        )
        return f0, voiced_flag

    # -----------------------------------------------------------
    # Step 1b: F0 Pitch Variation (Coefficient of Variation)
    #   Humans have rich, natural pitch movement when speaking.
    #   AI TTS tends to produce flatter, more monotone F0 contours.
    # -----------------------------------------------------------
    def measure_f0_variation(self, f0):
        if f0 is None:
            return 0.0
        voiced = f0[~np.isnan(f0)]
        if len(voiced) < 10:
            return 0.0
        cv = np.std(voiced) / (np.mean(voiced) + 1e-8)
        return float(cv)

    # -----------------------------------------------------------
    # Step 2: Detect 8-12 Hz modulation (physiological tremor)
    # -----------------------------------------------------------
    def detect_physiological_tremor(self, audio):
        env = np.abs(audio)
        nyq = 0.5 * self.sr
        b, a = signal.butter(4, 50 / nyq, btype='low')
        env_smooth = signal.filtfilt(b, a, env)
        env_smooth = env_smooth - np.mean(env_smooth)
        
        f, Pxx = signal.welch(env_smooth, fs=self.sr, nperseg=min(4096, len(env_smooth)))
        
        # 8-12 Hz physiological tremor band
        tremor_mask = (f >= 8) & (f <= 12)
        if not np.any(tremor_mask) or np.sum(Pxx) == 0:
            return 0.0
            
        tremor_power = np.sum(Pxx[tremor_mask])
        return float(tremor_power / (np.sum(Pxx) + 1e-8))

    # -----------------------------------------------------------
    # Step 3: Check breathing pattern (0.1-0.5 Hz amplitude cycle)
    # -----------------------------------------------------------
    def check_breathing_pattern(self, audio):
        env = np.abs(audio)
        nyq = 0.5 * self.sr
        b, a = signal.butter(2, 2.0 / nyq, btype='low') # Heavy lowpass
        env_smooth = signal.filtfilt(b, a, env)
        env_smooth = env_smooth - np.mean(env_smooth)
        
        f, Pxx = signal.welch(env_smooth, fs=self.sr, nperseg=min(self.sr * 4, len(env_smooth)))
        
        # 0.1 - 0.5 Hz breathing band
        breath_mask = (f >= 0.1) & (f <= 0.5)
        if not np.any(breath_mask) or np.sum(Pxx) == 0:
            return 0.0
            
        breath_power = np.sum(Pxx[breath_mask])
        return float(breath_power / (np.sum(Pxx) + 1e-8))


    # -----------------------------------------------------------
    # Step 4: Look for pre-speech activation (neural precursor)
    # -----------------------------------------------------------
    def look_for_prespeech_activation(self, audio):
        frame_len = 1024
        hop = 256

        energy = np.array([np.sum(audio[i:i+frame_len]**2) for i in range(0, len(audio)-frame_len, hop)])
        if len(energy) < 10:
            return 0.0

        energy = energy / (np.max(energy) + 1e-8)
        onset_indices = np.where(energy > 0.05)[0]
        
        if len(onset_indices) == 0:
            return 0.0

        onset = onset_indices[0]
        start = max(0, onset - 10) # Look slightly further back

        ramp = energy[start:onset]
        if len(ramp) < 2:
            return 0.0
            
        # Neural precursors exhibit a smooth exponential ramp rather than a sharp vertical step
        ramp_std = np.std(np.diff(ramp))
        if ramp_std < 1e-4: # Too smooth (AI often has 0 ramp or perfectly linear zeros)
            return 0.0
            
        ramp_smoothness = 1.0 / (ramp_std + 1e-8)
        return float(min(ramp_smoothness / 100.0, 1.0))  # Normalize to 0-1


    # -----------------------------------------------------------
    # Step 5: Measure natural vocal fold variation (jitter/shimmer)
    # -----------------------------------------------------------
    def measure_vocal_variation(self, audio, f0):
        if f0 is None:
            return 0.0, 0.0
            
        voiced = f0[~np.isnan(f0)]
        if len(voiced) < 10:
            return 0.0, 0.0

        # Jitter: cycle-to-cycle frequency variation
        diffs_f0 = np.abs(np.diff(voiced))
        jitter = np.mean(diffs_f0) / (np.mean(voiced) + 1e-8)
        
        # Shimmer: cycle-to-cycle amplitude variation
        # Rough approximation using energy contour
        frame_len = 1024
        hop = 256
        energy = np.array([np.sum(audio[i:i+frame_len]**2) for i in range(0, len(audio)-frame_len, hop)])
        
        # Match lengths roughly
        min_len = min(len(energy), len(voiced))
        if min_len < 5:
            return float(jitter), 0.0
            
        voiced_energy = energy[:min_len]
        diffs_env = np.abs(np.diff(voiced_energy))
        shimmer = np.mean(diffs_env) / (np.mean(voiced_energy) + 1e-8)

        return float(jitter), float(shimmer)


    # -----------------------------------------------------------
    # Step 6: Verify response timing (cognitive-motor delay)
    # -----------------------------------------------------------
    def verify_response_timing(self, audio):
        # Measure duration of initial silence
        frame_len = 1024
        hop = 256
        energy = np.array([np.sum(audio[i:i+frame_len]**2) for i in range(0, len(audio)-frame_len, hop)])
        
        if len(energy) < 10:
            return 0.0
            
        energy_norm = energy / (np.max(energy) + 1e-8)
        onset_indices = np.where(energy_norm > 0.05)[0]
        
        if len(onset_indices) == 0:
            return 0.0
            
        onset_frame = onset_indices[0]
        onset_time = librosa.frames_to_time(onset_frame, sr=self.sr, hop_length=hop)
        
        # Humans take 200ms - 1200ms to respond to mic prompts
        if 0.15 <= onset_time <= 1.5:
            delay_score = 1.0 - abs(onset_time - 0.6) / 1.0
            delay_score = max(delay_score, 0.3)
        else:
            delay_score = 0.05
            
        return float(delay_score)

    # -----------------------------------------------------------
    # Step 7: Spectral Flatness (Wiener Entropy)
    #   Real mic recordings contain natural breath noise and room
    #   ambience giving higher spectral flatness. AI TTS produces
    #   cleaner, more tonal spectra with lower flatness.
    # -----------------------------------------------------------
    def measure_spectral_flatness(self, audio):
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=2048, hop_length=512)
        return float(np.mean(flatness))

    # -----------------------------------------------------------
    # Step 8: Silence Noise Floor
    #   Real mic recordings have ambient room noise even in
    #   "silent" portions. AI files have near-zero silence.
    # -----------------------------------------------------------
    def measure_silence_noise_floor(self, audio):
        frame_len = 1024
        hop = 256
        rms = np.array([
            np.sqrt(np.mean(audio[i:i+frame_len]**2))
            for i in range(0, len(audio)-frame_len, hop)
        ])
        if len(rms) < 5:
            return 0.0
        
        # Find the quietest 20% of frames ("silence")
        threshold = np.percentile(rms, 20)
        silent_frames = rms[rms <= threshold]
        
        if len(silent_frames) == 0:
            return 0.0
        
        # Average noise floor during silence
        noise_floor = np.mean(silent_frames)
        return float(noise_floor)

    # -----------------------------------------------------------
    # Step 9: High-Frequency Noise Ratio
    #   Microphones inject self-noise spread across all frequencies.
    #   AI synthesis is spectrally clean above the voice band.
    # -----------------------------------------------------------
    def measure_high_freq_noise(self, audio):
        D = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        
        high_mask = freqs >= 6000  # Above voice band
        if not np.any(high_mask):
            return 0.0
        
        high_energy = np.mean(D[high_mask, :])
        total_energy = np.mean(D) + 1e-8
        
        return float(high_energy / total_energy)


    # -----------------------------------------------------------
    # Step 10: Spectral Consistency (frame-to-frame)
    #   AI TTS produces unnaturally uniform spectral shapes from
    #   frame to frame. Human speech has more irregular spectral
    #   variation due to articulatory movements + mic noise.
    # -----------------------------------------------------------
    def measure_spectral_consistency(self, audio):
        D = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        if D.shape[1] < 5:
            return 0.0

        # Normalize each frame to a probability distribution
        D_norm = D / (np.sum(D, axis=0, keepdims=True) + 1e-8)

        # Cosine similarity between consecutive frames
        similarities = []
        for i in range(D_norm.shape[1] - 1):
            a, b = D_norm[:, i], D_norm[:, i + 1]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            similarities.append(cos_sim)

        # High mean → very consistent → likely AI
        return float(np.mean(similarities))


    # -----------------------------------------------------------
    # FINAL ANALYSIS — Discriminative Scoring
    #
    # Instead of purely additive weights, each feature votes
    # FOR human origin or AGAINST it (evidence of AI synthesis).
    # Final decision = human_evidence − ai_evidence > 0
    #
    # Calibration (real tests):
    #   AI file:   noise_floor=0.003, jitter=0.010, f0_var=0.145
    #   Human mic: noise_floor>0.005, jitter>0.02,  f0_var>0.10
    # -----------------------------------------------------------
    def analyze(self, audio):
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # --- Extract all features ---
        f0, _ = self.extract_f0(audio)
        f0_variation = self.measure_f0_variation(f0)
        tremor = self.detect_physiological_tremor(audio)
        breathing = self.check_breathing_pattern(audio)
        prespeech = self.look_for_prespeech_activation(audio)
        jitter, shimmer = self.measure_vocal_variation(audio, f0)
        timing = self.verify_response_timing(audio)
        spectral_flatness = self.measure_spectral_flatness(audio)
        noise_floor = self.measure_silence_noise_floor(audio)
        hf_noise = self.measure_high_freq_noise(audio)
        spectral_consistency = self.measure_spectral_consistency(audio)

        # --- Discriminative scoring ---
        # Calibrated against real test data (5 human mic runs, multiple AI files):
        #
        # Strongest separators:
        #   prespeech:  AI=0.0 (no neural ramp), humans=0.82-1.0
        #   spec_flat:  AI<0.02 (pure synthesis), humans=0.051-0.078
        #   hf_noise:   AI=0.131-0.269, humans=0.030-0.129 (partial overlap)
        #
        # Removed from ai_evidence (overlap):
        #   timing < 0.65: human 5 scored 0.352, causes false negatives
        #   noise_floor: both cluster 0.002-0.011 after normalization
        # ------------------------------------------------------------
        human_evidence = 0.0
        ai_evidence = 0.0

        # ---- HF Noise (primary separator) ----
        # AI TTS files (especially MP3-encoded) have unusually HIGH hf_noise
        # (≈0.25-0.30) from codec quantization artifacts. Human mic recordings
        # have moderate hf_noise (0.03-0.15) from natural mic self-noise.
        if hf_noise > 0.18:
            ai_evidence += 35       # strongly AI-like (codec artifact pattern)
        elif hf_noise > 0.04:
            human_evidence += 30    # natural mic self-noise pattern
        elif hf_noise < 0.025:
            ai_evidence += 10       # suspiciously clean (lossless AI synthesis)

        # ---- Pre-speech Activation (strongest separator) ----
        # Human neural motor planning produces a smooth exponential energy
        # ramp before the first voiced frame. AI synthesis starts instantly
        # with no ramp — prespeech score is 0.0 or near-zero.
        if prespeech < 0.10:
            ai_evidence += 30       # no ramp → synthesis
        elif prespeech > 0.70:
            human_evidence += 20    # smooth neural ramp → human

        # ---- Cognitive-motor Timing (supporting, human-only reward) ----
        # Only reward clearly human timing; don't penalize mid-range values
        # since human onset can legitimately vary from 0.3–0.9s.
        if timing > 0.75:
            human_evidence += 15    # natural human response delay

        # ---- Jitter (supporting, only reward high jitter) ----
        # Humans can have low jitter too so we don't penalize low jitter,
        # but clearly high jitter is a human signal.
        if jitter > 0.016:
            human_evidence += 15

        # ---- F0 Variation (supporting) ----
        if f0_variation > 0.15:
            human_evidence += 10
        elif f0_variation < 0.04:
            ai_evidence += 10

        # ---- Spectral Flatness (strong AI indicator when very low) ----
        # AI synthesis produces unnaturally tonal/pure audio — very low
        # Wiener entropy. Human mic recordings always exceed 0.05.
        if spectral_flatness > 0.05:
            human_evidence += 15    # natural mic noise floor
        elif spectral_flatness < 0.03:
            ai_evidence += 25       # suspiciously pure — synthetic origin

        # ---- Physiological Tremor (light) ----
        if tremor > 0.06:
            human_evidence += 5

        # ---- Breathing Pattern (light) ----
        if breathing > 0.3:
            human_evidence += 5

        # ---- Shimmer (light) ----
        if shimmer > 0.22:
            human_evidence += 5

        liveness_score = human_evidence - ai_evidence
        is_human = liveness_score > 0

        return {
            "f0_variation": f0_variation,
            "physiological_tremor": tremor,
            "breathing_cycle": breathing,
            "prespeech_activation": prespeech,
            "jitter": jitter,
            "shimmer": shimmer,
            "cognitive_timing": timing,
            "spectral_flatness": spectral_flatness,
            "noise_floor": noise_floor,
            "hf_noise": hf_noise,
            "spectral_consistency": spectral_consistency,
            "human_evidence": human_evidence,
            "ai_evidence": ai_evidence,
            "liveness_score": liveness_score,
            "is_human": bool(is_human)
        }