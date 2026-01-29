"""
audio_processing.py

Advanced audio pipeline for drone sound detection.
Uses bandpass filtering + three feature extractors (MFCC, Autocorrelation, Cepstrum)
then classifies with a pre-trained SVM.

Dependencies:
    pip install librosa numpy scipy scikit-learn joblib
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt, correlate
from sklearn.svm import SVC
import joblib
import os

# ────────────────────────────────────────────────
# CONFIG (can be moved to config.py later)
# ────────────────────────────────────────────────
SAMPLE_RATE = 22050
BAND_LOW = 80.0
BAND_HIGH = 300.0
N_MFCC = 13
MODEL_PATH = "models/drone_audio_clf.pkl"

# ────────────────────────────────────────────────
# 1. Bandpass Filter (80–300 Hz)
# ────────────────────────────────────────────────
def bandpass_filter(data, lowcut=BAND_LOW, highcut=BAND_HIGH, fs=SAMPLE_RATE, order=4):
    """
    Butterworth bandpass filter to isolate drone propeller frequency range.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# ────────────────────────────────────────────────
# 2. Feature Extraction: MFCC, Autocorrelation, Cepstrum
# ────────────────────────────────────────────────
def extract_features(audio_chunk, sr=SAMPLE_RATE):
    """
    Extract robust features from a filtered audio chunk.
    Returns a 1D numpy array of 17 features.
    """
    # 2.1 MFCC (timbre / spectral shape)
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfccs, axis=1)                     # shape: (13,)

    # 2.2 Autocorrelation (mechanical periodicity)
    autocorr = correlate(audio_chunk, audio_chunk, mode='full')
    autocorr = autocorr[len(autocorr)//2:]                 # positive lags only
    autocorr = autocorr / (np.max(autocorr) + 1e-8)        # normalize
    # Find first significant peak after lag=10 (skip DC)
    peak_idx = np.argmax(autocorr[10:]) + 10 if len(autocorr) > 10 else 0
    autocorr_strength = autocorr[peak_idx]
    autocorr_period = peak_idx / sr                        # period in seconds
    autocorr_feats = np.array([autocorr_strength, autocorr_period])

    # 2.3 Cepstrum (harmonic spacing / rahmonics)
    spectrum = np.abs(np.fft.fft(audio_chunk))
    log_spectrum = np.log(spectrum + 1e-8)
    cepstrum = np.abs(np.fft.ifft(log_spectrum))
    cepstrum = cepstrum[:len(cepstrum)//2]                 # positive quefrency
    # Find first significant rahmonic (skip DC)
    peak_idx = np.argmax(np.abs(cepstrum[5:])) + 5 if len(cepstrum) > 5 else 0
    cep_strength = np.max(np.abs(cepstrum))
    cep_quefrency = peak_idx / sr                          # quefrency in seconds
    cep_feats = np.array([cep_strength, cep_quefrency])

    # Combine all features into one vector
    features = np.concatenate([
        mfcc_mean,              # 13
        autocorr_feats,         # 2
        cep_feats               # 2
    ])                          # Total: 17 features

    return features

# ────────────────────────────────────────────────
# 3. Load Pre-trained Classifier
# ────────────────────────────────────────────────
def load_audio_classifier(model_path=MODEL_PATH):
    """
    Load the trained SVM/RF classifier.
    Returns the classifier object or raises FileNotFoundError.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Audio classifier not found at {model_path}. Train it first.")
    return joblib.load(model_path)

# ────────────────────────────────────────────────
# 4. Main Audio Processing Function (called per synced chunk)
# ────────────────────────────────────────────────
def process_audio_chunk(chunk, sr=SAMPLE_RATE, clf=None):
    """
    Full pipeline for one audio chunk.
    Returns audio_conf (0–1) — 0 means "not drone-like"
    """
    if clf is None:
        clf = load_audio_classifier()

    # Step 1: Bandpass filter
    filtered = bandpass_filter(chunk, fs=sr)

    # Step 2: Extract features
    features = extract_features(filtered, sr=sr)

    # Step 3: Classify
    prob = clf.predict_proba([features])[0][1]  # probability of "drone" class
    audio_conf = prob if prob > 0.55 else 0.0   # threshold (tune this)

    return audio_conf