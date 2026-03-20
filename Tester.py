import librosa
import numpy as np
import pandas as pd
import xgboost as xgb

# === FEATURE EXTRACTION FUNCTION ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Harmonic features (simplified proxies)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=8000)
    fundamental_freq = np.nanmean(f0)
    num_harmonic_peaks = np.count_nonzero(~np.isnan(f0))
    harmonic_spacing_std = np.nanstd(f0)
    harmonic_regularity = np.nanmean(voiced_probs)
    harmonic_power_ratio = np.mean(librosa.feature.rms(y=y))

    # Peak prominence (proxy: max amplitude vs mean)
    peak_prominence_mean = np.max(np.abs(y)) - np.mean(np.abs(y))

    # Spectral features
    spectral_flatness_mean = np.mean(librosa.feature.spectral_flatness(y=y))
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_spread = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_contrast_mean = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # RMS features
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    rms_stability = rms_std / (rms_mean + 1e-6)

    # Zero crossing
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    # Frequency variance
    freq_variance_mean = np.var(librosa.stft(y))

    # Band energies
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    def band_energy(low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.sum(S[mask])

    total_energy = np.sum(S)
    energy_low_freq = band_energy(0, 500)
    energy_mid_freq = band_energy(500, 2000)
    energy_high_freq = band_energy(2000, 6000)
    energy_vhigh_freq = band_energy(6000, 12000)

    low_to_total_ratio = energy_low_freq / (total_energy + 1e-6)
    mid_to_total_ratio = energy_mid_freq / (total_energy + 1e-6)
    high_to_total_ratio = energy_high_freq / (total_energy + 1e-6)
    vhigh_to_total_ratio = energy_vhigh_freq / (total_energy + 1e-6)

    # Horizontal band strength (proxy: spectral bandwidth)
    horizontal_band_strength = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Temporal correlation (proxy: autocorrelation mean)
    autocorr = np.correlate(y, y, mode='full')
    temporal_correlation_mean = np.mean(autocorr)

    # Assemble features in exact order
    features = {
        "num_harmonic_peaks": num_harmonic_peaks,
        "fundamental_freq": fundamental_freq,
        "harmonic_spacing_std": harmonic_spacing_std,
        "harmonic_regularity": harmonic_regularity,
        "harmonic_power_ratio": harmonic_power_ratio,
        "peak_prominence_mean": peak_prominence_mean,
        "spectral_flatness_mean": spectral_flatness_mean,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_spread": spectral_spread,
        "spectral_contrast_mean": spectral_contrast_mean,
        "spectral_rolloff": spectral_rolloff,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_stability": rms_stability,
        "zero_crossing_rate": zero_crossing_rate,
        "freq_variance_mean": freq_variance_mean,
        "horizontal_band_strength": horizontal_band_strength,
        "temporal_correlation_mean": temporal_correlation_mean,
        "energy_low_freq": energy_low_freq,
        "low_to_total_ratio": low_to_total_ratio,
        "energy_mid_freq": energy_mid_freq,
        "mid_to_total_ratio": mid_to_total_ratio,
        "energy_high_freq": energy_high_freq,
        "high_to_total_ratio": high_to_total_ratio,
        "energy_vhigh_freq": energy_vhigh_freq,
        "vhigh_to_total_ratio": vhigh_to_total_ratio,
    }

    return pd.DataFrame([features])

# === LOAD MODEL ===
model = xgb.XGBClassifier()
model.load_model(r"C:\Users\Omprakash\Desktop\pthon\drone_detector_xgb.model")

# === PREDICT ON NEW AUDIO ===
file_path = r"C:\Users\Omprakash\Desktop\pthon\testoutput\traffic.wav"   # replace with your audio file
X_new = extract_features(file_path)

y_prob = model.predict_proba(X_new)[:, 1]

# Custom threshold
threshold = 0.35
y_pred_custom = (y_prob > threshold).astype(int)

# Final result
print("Final Prediction:", "Drone" if y_pred_custom[0] == 1 else "Background")
print("Probability of Drone:", y_prob[0])