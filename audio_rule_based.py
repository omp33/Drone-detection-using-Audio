# audio_rule_based.py
# -------------------------------------------------
# Rule-based drone audio presence detector
# Uses:
# - Multi-band spectral energy
# - Frequency bin contiguity
# - Temporal consistency
# -------------------------------------------------

import numpy as np
import librosa
from collections import deque
HISTORY_LEN = 3
REQUIRED_HITS = 2

band_history = deque(maxlen=HISTORY_LEN)

SAMPLE_RATE = 22050

# Frequency bands to scan (Hz)
FREQ_BANDS = [
    (80, 400),
    (300, 900),
    (800, 2000)
]

# Tunable thresholds (reasonable defaults)
ENERGY_RATIO_TH = 0.15     # band energy / total energy
BIN_CONTIGUITY_TH = 4      # min consecutive freq bins
TEMPORAL_FRAMES_TH = 3     # frames must persist
FRAME_SEC = 0.1            # STFT frame duration


# -------------------------------------------------
def detect_drone_audio(y, sr=SAMPLE_RATE, debug=False):
    """
    Rule-based drone audio presence detector.
    Returns structured result, not fake probabilities.
    """

    if len(y) < int(0.5 * sr):
        return {"audio_present": False, "reason": "too_short"}

    # STFT
    n_fft = 2048
    hop_length = int(FRAME_SEC * sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    total_energy = np.sum(S) + 1e-9

    band_hits = []

    for (f_low, f_high) in FREQ_BANDS:
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        if len(idx) == 0:
            continue

        band_energy = np.sum(S[idx, :])
        energy_ratio = band_energy / total_energy

        if energy_ratio < ENERGY_RATIO_TH:
            continue

        # ---- Frequency bin contiguity ----
        active_bins = np.mean(S[idx, :], axis=1)
        active_mask = active_bins > (0.3 * np.max(active_bins))

        max_run = _max_consecutive(active_mask)

        if max_run < BIN_CONTIGUITY_TH:
            continue

        # ---- Temporal consistency ----
        temporal_energy = np.mean(S[idx, :], axis=0)
        temporal_mask = temporal_energy > (0.4 * np.max(temporal_energy))

        if _max_consecutive(temporal_mask) < TEMPORAL_FRAMES_TH:
            continue

        band_hits.append((f_low, f_high))

    result = {
        "audio_present": len(band_hits) >= 2,
        "bands_confirmed": band_hits,
        "strength": _strength_from_hits(len(band_hits))
    }

    if debug:
        print("[DEBUG] Confirmed bands:", band_hits)

    return result


# -------------------------------------------------
def _max_consecutive(mask):
    """Longest run of True values."""
    max_run = run = 0
    for v in mask:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _strength_from_hits(n):
    if n >= 3:
        return "strong"
    if n == 2:
        return "moderate"
    return "weak"
