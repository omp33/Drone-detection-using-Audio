import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# -----------------------------
# USER-TUNABLE VISUAL PARAMETERS
# -----------------------------
SPEC_N_FFT = 2048        # larger FFT for better frequency resolution in spectrogram
SPEC_HOP = 256           # hop length (smaller -> better time resolution)
N_MELS = 128             # number of mel bands (higher -> more vertical detail)
SPEC_VMIN_PCT = 1        # lower percentile for spectrogram contrast
SPEC_VMAX_PCT = 99       # upper percentile for spectrogram contrast
WAVEFORM_LINEWIDTH = 1.2
WAVEFORM_COLOR = "deepskyblue"
SPECTROGRAM_CMAP = "inferno"
SAVE_FIG = True          # set to True to save the figure
SAVE_PATH = "Near Drone.png"
SAVE_DPI = 300

# -----------------------------
# FEATURE FUNCTIONS
# -----------------------------

def harmonic_features(spectrum, freqs):
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.2)
    peak_freqs = freqs[peaks]
    peak_count = len(peak_freqs)
    regularity = 1 / (np.std(np.diff(peak_freqs)) + 1e-6) if peak_count > 2 else 0
    return peak_count, regularity

def low_frequency_ratio(spectrum, freqs):
    total = np.sum(spectrum)
    idx = np.logical_and(freqs >= 50, freqs <= 300)
    low_energy = np.sum(spectrum[idx])
    return low_energy / (total + 1e-6)

def periodicity_strength(signal):
    autocorr = librosa.autocorrelate(signal)
    autocorr = autocorr[len(autocorr)//2:]
    peak = np.max(autocorr)
    return peak / (np.sum(autocorr) + 1e-6)

def horizontal_band_strength(spec):
    mean_time = np.mean(spec, axis=1)
    return np.std(mean_time)

def spectral_features(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    return np.mean(centroid), np.std(centroid), np.mean(flatness)

# -----------------------------
# MAIN ANALYSIS
# -----------------------------

def analyze_audio(audio_file, save_path=SAVE_PATH, save_fig=SAVE_FIG):
    y, sr = librosa.load(audio_file, sr=None)

    # --- STFT and averaged spectrum (used for harmonic features) ---
    n_fft_spec = SPEC_N_FFT
    hop_spec = SPEC_HOP
    S = np.abs(librosa.stft(y, n_fft=n_fft_spec, hop_length=hop_spec))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft_spec)
    spectrum = np.mean(S, axis=1)

    # ---- Features ----
    peak_count, harmonic_regularity = harmonic_features(spectrum, freqs)
    low_ratio = low_frequency_ratio(spectrum, freqs)
    periodicity = periodicity_strength(y)
    horizontal_strength = horizontal_band_strength(S)
    centroid_mean, centroid_std, flatness = spectral_features(y, sr)

    # ---- Log Mel Spectrogram (higher resolution) ----
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft_spec,
        hop_length=hop_spec,
        n_mels=N_MELS,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Compute robust vmin/vmax from percentiles to avoid washed-out or clipped images
    vmin = np.percentile(log_mel, SPEC_VMIN_PCT)
    vmax = np.percentile(log_mel, SPEC_VMAX_PCT)

    # -----------------------------
    # VISUALIZATION (3x2 layout preserved)
    # -----------------------------
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12))

    # Waveform (top-left)
    ax1 = plt.subplot(3, 2, 1)
    # Normalize for consistent amplitude display
    y_display = y / (np.max(np.abs(y)) + 1e-9)
    librosa.display.waveshow(y_display, sr=sr, color=WAVEFORM_COLOR, linewidth=WAVEFORM_LINEWIDTH, ax=ax1)
    ax1.set_title("Waveform", color="white", pad=12)   # pad to avoid overlap
    ax1.set_xlabel("")  # remove x label to reduce clutter
    ax1.set_ylabel("Amplitude", color="lightgray")

    # Log Mel Spectrogram (top-right)
    ax2 = plt.subplot(3, 2, 2)
    img = librosa.display.specshow(
    log_mel,
    sr=sr,
    hop_length=SPEC_HOP,
    x_axis="time",
    y_axis="mel",
    cmap=SPECTROGRAM_CMAP,
    vmin=vmin,
    vmax=vmax,
    ax=ax2
)
    ax2.set_title("Log Mel Spectrogram", color="white", pad=12)
    ax2.set_ylabel("Mel band", color="lightgray")
    ax2.set_aspect("auto")   # <-- set aspect here, not in specshow
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="lightgray")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="lightgray")



    # Harmonic spectrum (middle-left)
    ax3 = plt.subplot(3, 2, 3)
    
    ax3.plot(freqs, spectrum, color="lime")
    
    ax3.set_xlim(0, 3000)
    ax3.set_title("Harmonic Spectrum", color="white", pad=10)
    ax3.set_xlabel("Frequency (Hz)", color="lightgray")
    ax3.set_ylabel("Magnitude", color="lightgray")

    # Spectral centroid drift (middle-right)
    ax4 = plt.subplot(3, 2, 4)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times_cent = librosa.times_like(centroid, sr=sr, hop_length=hop_spec)
    ax4.plot(times_cent, centroid, color="gold")
    ax4.set_title("Spectral Centroid Drift", color="white", pad=10)
    ax4.set_xlabel("Time (s)", color="lightgray")
    ax4.set_ylabel("Hz", color="lightgray")

    # Band energy distribution (bottom-left)
    ax5 = plt.subplot(3, 2, 5)
    bands = {
        "Low (50-300Hz)": (50, 300),
        "Mid (300-1000Hz)": (300, 1000),
        "High (1000-2500Hz)": (1000, 2500)
    }
    band_energy = []
    for b in bands.values():
        idx = np.logical_and(freqs >= b[0], freqs <= b[1])
        # protect against empty idx
        band_energy.append(np.mean(spectrum[idx]) if np.any(idx) else 0.0)
    ax5.bar(list(bands.keys()), band_energy, color=["deepskyblue", "orange", "magenta"])
    ax5.set_title("Energy Distribution", color="white", pad=10)
    ax5.set_ylabel("Mean magnitude", color="lightgray")

    # Feature summary placeholder (bottom-right)
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis("off")
    ax6.text(0.5, 0.5, "Feature Summary Removed", ha="center", va="center", color="gray")

    # -----------------------------
    # Tidy layout and avoid title overlap
    # -----------------------------
    # Increase spacing between subplots and top margin to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.45, wspace=0.35, top=0.93)

    # Add a main title with some top margin
    fig.suptitle(os.path.basename(audio_file), color="white", fontsize=14, y=0.99)

    # Save figure if requested
    if save_fig:
        try:
            fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Saved figure to: {os.path.abspath(save_path)} (dpi={SAVE_DPI})")
        except Exception as e:
            print("Failed to save figure:", e)

    plt.show()

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio.py <audio_file> [optional_save_path]")
        sys.exit(1)

    audio_file = sys.argv[1]
    if len(sys.argv) >= 3:
        SAVE_PATH = sys.argv[2]

    analyze_audio(audio_file, save_path=SAVE_PATH, save_fig=SAVE_FIG)
