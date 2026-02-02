import numpy as np
import librosa
import matplotlib.pyplot as plt

# ============================
# CONFIG — TUNE THESE ONCE
# ============================

SAMPLE_RATE = 22050

N_FFT = 1024                 # small FFT → good time resolution
HOP_LENGTH = 256
FMIN = 80                    # ignore rumble
FMAX = 1200                  # drones live here
PERSIST_FRAMES = 12          # ~0.15–0.2s persistence
ENERGY_PERCENTILE = 75       # adaptive threshold
HARMONIC_TOL = 0.08          # 8% frequency tolerance

# ============================
# CORE LOGIC
# ============================

def compute_log_spectrogram(y, sr):
    S = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann"
    )
    S_mag = np.abs(S)
    S_log = librosa.amplitude_to_db(S_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    return S_log, freqs, times


def band_mask(freqs):
    return np.where((freqs >= FMIN) & (freqs <= FMAX))[0]


def persistence_score(S_band):
    """
    Measures how long energy stays 'on' in narrow frequency bins.
    """
    scores = []
    for f_bin in range(S_band.shape[0]):
        row = S_band[f_bin]
        thresh = np.percentile(row, ENERGY_PERCENTILE)
        active = row > thresh

        # count longest continuous run
        max_run = 0
        run = 0
        for v in active:
            if v:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0

        scores.append(max_run)

    return np.array(scores)


def harmonic_check(S_band, freqs_band):
    """
    Weak harmonic agreement check.
    """
    mean_energy = np.mean(S_band, axis=1)
    f0_idx = np.argmax(mean_energy)
    f0 = freqs_band[f0_idx]

    if f0 < 100:
        return False

    for mult in [2, 3]:
        target = mult * f0
        tol = target * HARMONIC_TOL
        if target > freqs_band[-1]:
            continue

        idx = np.where(np.abs(freqs_band - target) < tol)[0]
        if len(idx) == 0:
            return False

        if np.mean(mean_energy[idx]) < 0.3 * mean_energy[f0_idx]:
            return False

    return True


def detect_drone_audio(y, sr=SAMPLE_RATE, debug=False):
    S_log, freqs, times = compute_log_spectrogram(y, sr)
    band_idx = band_mask(freqs)

    S_band = S_log[band_idx, :]
    freqs_band = freqs[band_idx]

    pers = persistence_score(S_band)

    persistent_bins = pers >= PERSIST_FRAMES
    num_persistent = np.sum(persistent_bins)

    if num_persistent < 2:
        return {
            "audio_present": False,
            "strength": "none",
            "score": 0.0,
            "dominant_band": None
        }

    # Cluster persistent bins
    idxs = np.where(persistent_bins)[0]
    f_low = freqs_band[idxs[0]]
    f_high = freqs_band[idxs[-1]]

    harmonic_ok = harmonic_check(S_band, freqs_band)

    score = num_persistent
    if harmonic_ok:
        score *= 1.4

    if score > 20:
        strength = "strong"
    elif score > 10:
        strength = "medium"
    else:
        strength = "weak"

    if debug:
        plt.figure(figsize=(10,4))
        plt.imshow(
            S_band,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs_band[0], freqs_band[-1]],
            cmap="magma"
        )
        plt.colorbar(label="dB")
        plt.title("Log Spectrogram (Drone Band)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.show()

    return {
        "audio_present": True,
        "strength": strength,
        "score": float(score),
        "dominant_band": (float(f_low), float(f_high))
    }
