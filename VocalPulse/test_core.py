import numpy as np
from src.core import VocalPulseCore

core = VocalPulseCore()

# 3 seconds test signal at 16kHz
sr = 16000
t = np.linspace(0, 3, 3 * sr)

# Add 0.5s cognitive delay at start
t_delay = np.zeros(int(0.5 * sr))

# Organic voice base
base = np.sin(2 * np.pi * 150 * t[len(t_delay):])

# 1. Micro-jitter variation
jitter = 1.0 + 0.15 * np.random.randn(len(base))

# 2. Physiological tremor (10 Hz modulation)
tremor = 1.0 + 0.3 * np.sin(2 * np.pi * 10 * t[len(t_delay):])

# 3. Breathing envelope (0.3 Hz slow envelope)
breath = 1.0 + 0.2 * np.sin(2 * np.pi * 0.3 * t[len(t_delay):])

# 4. Neural-prespeech ramp (soft exponential onset)
ramp = np.linspace(0, 1, int(0.1 * sr)) ** 2
envelope = np.ones(len(base))
envelope[:len(ramp)] = ramp

speech = base * jitter * tremor * breath * envelope

# Combine delay and speech
signal = np.concatenate((t_delay, speech))

result = core.analyze(signal)

print("F0 Variation:", result["f0_variation"])
print("Tremor Score:", result["physiological_tremor"])
print("Breathing Cycle:", result["breathing_cycle"])
print("Jitter:", result["jitter"])
print("Spectral Flatness:", result["spectral_flatness"])
print("Liveness Score:", result["liveness_score"])
print("Is Human:", result["is_human"])