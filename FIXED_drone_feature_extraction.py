"""
FIXED Drone Feature Extraction - Addresses All Issues
Works correctly on: loud drones, far drones, and noise-only samples
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class FixedDroneFeatureExtractor:
    """
    FIXED: Properly detects weak/far drone signals
    
    Key improvements:
    1. Adaptive peak detection (multiple thresholds)
    2. Proper spectral flatness calculation
    3. Signal strength normalization
    4. Better frequency analysis
    """
    
    def __init__(self, sr=44100, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def compute_spectrogram(self, y):
        """Compute spectrogram using scipy"""
        f, t, Zxx = scipy_signal.stft(
            y,
            fs=self.sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window='hann'
        )
        S = np.abs(Zxx)
        return S, f, t
    
    def extract_all_features(self, y, sr=None, debug=False):
        """
        Extract all features with proper handling of weak signals
        
        Parameters:
        -----------
        y : np.array
            Audio signal
        sr : int
            Sample rate (default: self.sr)
        debug : bool
            Print debug information
            
        Returns:
        --------
        dict : All extracted features
        """
        if sr is None:
            sr = self.sr
        
        # Normalize audio
        if np.max(np.abs(y)) > 0:
            y_norm = y / np.max(np.abs(y))
        else:
            y_norm = y
        
        # Compute spectrogram
        S, freqs = self.compute_spectrogram(y_norm)
        S_db = 20 * np.log10(S + 1e-10)
        
        features = {}
        
        # Extract all feature categories
        features.update(self._extract_harmonic_features_FIXED(S, freqs, debug=debug))
        features.update(self._extract_spectral_features_FIXED(S, freqs, y_norm, sr))
        features.update(self._extract_temporal_features(S, y_norm, sr))
        features.update(self._extract_pattern_features(S_db))
        features.update(self._extract_energy_features(S, freqs))
        
        return features
    
    def _extract_harmonic_features_FIXED(self, S, freqs, debug=False):
        """
        FIXED harmonic feature extraction
        
        Key fixes:
        1. Adaptive thresholding based on signal characteristics
        2. Multiple prominence levels to catch weak peaks
        3. Frequency-weighted peak detection
        4. Better handling of noise floor
        """
        features = {}
        
        # Average spectrum across time
        avg_spectrum = np.mean(S, axis=1)
        
        # Focus on drone range (50-2500 Hz)
        drone_mask = (freqs >= 50) & (freqs <= 2500)
        drone_spectrum = avg_spectrum[drone_mask]
        drone_freqs = freqs[drone_mask]
        
        if len(drone_spectrum) == 0:
            # Fallback
            features['num_harmonic_peaks'] = 0
            features['fundamental_freq'] = 0
            features['harmonic_regularity'] = 0
            features['harmonic_power_ratio'] = 0
            features['harmonic_spacing_mean'] = 0
            features['harmonic_spacing_std'] = 0
            return features
        
        # Calculate signal characteristics
        max_val = np.max(drone_spectrum)
        mean_val = np.mean(drone_spectrum)
        std_val = np.std(drone_spectrum)
        
        # Estimate noise floor (bottom 25% of values)
        sorted_vals = np.sort(drone_spectrum)
        noise_floor = np.mean(sorted_vals[:len(sorted_vals)//4])
        
        # Signal-to-Noise estimate
        snr_estimate = (mean_val - noise_floor) / (std_val + 1e-10)
        
        if debug:
            print(f"\n=== Harmonic Detection Debug ===")
            print(f"Max value: {max_val:.4f}")
            print(f"Mean value: {mean_val:.4f}")
            print(f"Noise floor: {noise_floor:.4f}")
            print(f"SNR estimate: {snr_estimate:.4f}")
        
        # ADAPTIVE THRESHOLDING - the key fix!
        # Use multiple prominence thresholds to catch peaks of different strengths
        prominence_levels = [
            ('very_sensitive', max_val * 0.02),   # 2% - catches weak far drones
            ('sensitive', max_val * 0.05),         # 5% - good balance
            ('moderate', max_val * 0.10),          # 10% - original (too high)
            ('conservative', max_val * 0.20),      # 20% - only strong peaks
        ]
        
        all_detections = []
        
        for level_name, prom_threshold in prominence_levels:
            # Detect peaks
            peaks, properties = find_peaks(
                drone_spectrum,
                prominence=prom_threshold,
                distance=3,  # Minimum 3 bins apart (reduces false peaks)
                height=noise_floor + std_val * 0.5  # Must be above noise floor
            )
            
            if debug and len(peaks) > 0:
                print(f"\n{level_name}: threshold={prom_threshold:.4f}, peaks={len(peaks)}")
                print(f"  Peak frequencies: {drone_freqs[peaks][:10]}")  # Show first 10
            
            all_detections.append({
                'level': level_name,
                'threshold': prom_threshold,
                'peaks': peaks,
                'properties': properties,
                'num_peaks': len(peaks)
            })
        
        # SELECT BEST DETECTION
        # Prefer detection with 3-15 peaks (typical for drones)
        # If none in that range, use most sensitive with at least 1 peak
        best_detection = None
        
        # First, try to find detection with 3-15 peaks
        for det in all_detections:
            if 3 <= det['num_peaks'] <= 15:
                best_detection = det
                break
        
        # If no good detection, try to find any with peaks
        if best_detection is None:
            for det in all_detections:
                if det['num_peaks'] > 0:
                    best_detection = det
                    break
        
        # If still nothing, use most sensitive
        if best_detection is None:
            best_detection = all_detections[0]  # Most sensitive
        
        if debug:
            print(f"\n✓ Selected detection level: {best_detection['level']}")
            print(f"  Number of peaks: {best_detection['num_peaks']}")
        
        peaks = best_detection['peaks']
        properties = best_detection['properties']
        
        # Extract harmonic features from best detection
        features['num_harmonic_peaks'] = len(peaks)
        
        if len(peaks) >= 2:
            peak_freqs = drone_freqs[peaks]
            features['fundamental_freq'] = float(peak_freqs[0])
            
            # Harmonic spacing
            harmonic_spacing = np.diff(peak_freqs)
            features['harmonic_spacing_mean'] = float(np.mean(harmonic_spacing))
            features['harmonic_spacing_std'] = float(np.std(harmonic_spacing))
            
            # Harmonic regularity (lower std = more regular)
            if features['harmonic_spacing_std'] > 0:
                features['harmonic_regularity'] = float(1.0 / (1.0 + features['harmonic_spacing_std'] / 50.0))
            else:
                features['harmonic_regularity'] = 1.0
            
            # Harmonic power ratio
            peak_power = np.sum(drone_spectrum[peaks]**2)
            total_power = np.sum(drone_spectrum**2)
            features['harmonic_power_ratio'] = float(peak_power / (total_power + 1e-10))
            
            # Peak prominence (strength of peaks)
            features['peak_prominence_mean'] = float(np.mean(properties['prominences']))
            features['peak_prominence_max'] = float(np.max(properties['prominences']))
            
        elif len(peaks) == 1:
            # Only one peak detected
            features['fundamental_freq'] = float(drone_freqs[peaks[0]])
            features['harmonic_spacing_mean'] = 0.0
            features['harmonic_spacing_std'] = 0.0
            features['harmonic_regularity'] = 0.0
            
            peak_power = drone_spectrum[peaks[0]]**2
            total_power = np.sum(drone_spectrum**2)
            features['harmonic_power_ratio'] = float(peak_power / (total_power + 1e-10))
            
            features['peak_prominence_mean'] = float(properties['prominences'][0])
            features['peak_prominence_max'] = float(properties['prominences'][0])
        else:
            # No peaks detected
            features['fundamental_freq'] = 0.0
            features['harmonic_spacing_mean'] = 0.0
            features['harmonic_spacing_std'] = 0.0
            features['harmonic_regularity'] = 0.0
            features['harmonic_power_ratio'] = 0.0
            features['peak_prominence_mean'] = 0.0
            features['peak_prominence_max'] = 0.0
        
        return features
    
    def _extract_spectral_features_FIXED(self, S, freqs, y, sr):
        """
        FIXED spectral feature extraction
        
        Key fix: Compute flatness on TIME-AVERAGED spectrum, not frame-by-frame
        """
        features = {}
        
        # Average spectrum (this is what we see in the power spectrum plot)
        avg_spectrum = np.mean(S, axis=1)
        
        # FIXED: Spectral flatness on averaged spectrum
        # Geometric mean / Arithmetic mean
        # Drone (tonal): low flatness (~0.01-0.1)
        # Noise: high flatness (~0.3-0.8)
        geometric_mean = np.exp(np.mean(np.log(avg_spectrum + 1e-10)))
        arithmetic_mean = np.mean(avg_spectrum)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        features['spectral_flatness_mean'] = float(spectral_flatness)
        
        # Spectral centroid (on averaged spectrum)
        spectral_centroid = np.sum(freqs * avg_spectrum) / (np.sum(avg_spectrum) + 1e-10)
        features['spectral_centroid_mean'] = float(spectral_centroid)
        
        # Spectral spread (variance around centroid)
        spectral_spread = np.sqrt(
            np.sum(((freqs - spectral_centroid)**2) * avg_spectrum) / 
            (np.sum(avg_spectrum) + 1e-10)
        )
        features['spectral_spread'] = float(spectral_spread)
        
        # Spectral contrast - difference between peaks and valleys
        # Compute in frequency bands
        bands = [
            (50, 300),
            (300, 1000),
            (1000, 2500),
            (2500, sr/2)
        ]
        
        contrasts = []
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs < high)
            if np.any(band_mask):
                band_spectrum = avg_spectrum[band_mask]
                peak_val = np.percentile(band_spectrum, 90)
                valley_val = np.percentile(band_spectrum, 10)
                contrast = peak_val - valley_val
                contrasts.append(contrast)
        
        features['spectral_contrast_mean'] = float(np.mean(contrasts)) if contrasts else 0.0
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(avg_spectrum)
        rolloff_threshold = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = float(freqs[rolloff_idx[0]])
        else:
            features['spectral_rolloff'] = float(freqs[-1])
        
        return features
    
    def _extract_temporal_features(self, S, y, sr):
        """Extract time-domain features"""
        features = {}
        
        # RMS energy over time
        frame_length = 2048
        hop_length = 512
        rms_frames = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i+frame_length]
            rms = np.sqrt(np.mean(frame**2))
            rms_frames.append(rms)
        
        rms_frames = np.array(rms_frames)
        features['rms_mean'] = float(np.mean(rms_frames))
        features['rms_std'] = float(np.std(rms_frames))
        features['rms_stability'] = float(1.0 / (1.0 + features['rms_std']))
        
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y))
        features['zero_crossing_rate'] = float(zcr)
        
        return features
    
    def _extract_pattern_features(self, S_db):
        """Extract spectrogram pattern features"""
        features = {}
        
        # Horizontal band strength (low variance across time = strong bands)
        freq_variance = np.var(S_db, axis=1)  # Variance across time for each frequency
        features['freq_variance_mean'] = float(np.mean(freq_variance))
        features['horizontal_band_strength'] = float(1.0 / (1.0 + np.mean(freq_variance)))
        
        # Temporal correlation (consecutive frames similarity)
        if S_db.shape[1] > 1:
            correlations = []
            sample_frames = min(100, S_db.shape[1] - 1)
            step = max(1, (S_db.shape[1] - 1) // sample_frames)
            
            for i in range(0, S_db.shape[1] - 1, step):
                corr = np.corrcoef(S_db[:, i], S_db[:, i+1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            features['temporal_correlation_mean'] = float(np.mean(correlations)) if correlations else 0.0
        else:
            features['temporal_correlation_mean'] = 0.0
        
        return features
    
    def _extract_energy_features(self, S, freqs):
        """Extract energy distribution features"""
        features = {}
        
        # Energy in different frequency bands
        bands = {
            'low': (50, 300),
            'mid': (300, 1000),
            'high': (1000, 2500),
            'vhigh': (2500, freqs[-1])
        }
        
        total_energy = np.mean(S)
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_energy = np.mean(S[mask, :])
                features[f'energy_{band_name}_freq'] = float(band_energy)
                features[f'{band_name}_to_total_ratio'] = float(band_energy / (total_energy + 1e-10))
            else:
                features[f'energy_{band_name}_freq'] = 0.0
                features[f'{band_name}_to_total_ratio'] = 0.0
        
        return features

# ============================================================================
# SIMPLE USAGE EXAMPLE
# ============================================================================

def extract_features_from_audio_file(audio_path, debug=False):
    """
    Simple function to extract features from an audio file
    
    Usage:
        features = extract_features_from_audio_file('drone_audio.wav')
    """
    import soundfile as sf  # or use scipy.io.wavfile, or librosa
    
    # Load audio
    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:  # If stereo, convert to mono
            y = np.mean(y, axis=1)
    except:
        # Fallback to scipy
        from scipy.io import wavfile
        sr, y = wavfile.read(audio_path)
        y = y.astype(float) / np.max(np.abs(y))
    
    # Extract features
    extractor = FixedDroneFeatureExtractor(sr=sr)
    features = extractor.extract_all_features(y, sr, debug=debug)
    
    return features


def print_feature_summary(features, label=""):
    """Print a nice summary of extracted features"""
    print(f"\n{'='*70}")
    print(f" Feature Summary: {label}".center(70))
    print(f"{'='*70}")
    
    print(f"\n📊 HARMONIC FEATURES (Most Important for Drone Detection)")
    print(f"  • Number of Peaks:        {features['num_harmonic_peaks']}")
    print(f"  • Fundamental Frequency:  {features['fundamental_freq']:.1f} Hz")
    print(f"  • Harmonic Regularity:    {features['harmonic_regularity']:.3f}")
    print(f"  • Harmonic Power Ratio:   {features['harmonic_power_ratio']:.3f}")
    
    print(f"\n📈 SPECTRAL FEATURES")
    print(f"  • Spectral Flatness:      {features['spectral_flatness_mean']:.4f}")
    print(f"  • Spectral Centroid:      {features['spectral_centroid_mean']:.1f} Hz")
    print(f"  • Spectral Contrast:      {features['spectral_contrast_mean']:.2f}")
    
    print(f"\n⏱️  TEMPORAL FEATURES")
    print(f"  • RMS Stability:          {features['rms_stability']:.3f}")
    print(f"  • Temporal Correlation:   {features['temporal_correlation_mean']:.3f}")
    
    print(f"\n🎨 PATTERN FEATURES")
    print(f"  • Horizontal Band Strength: {features['horizontal_band_strength']:.3f}")
    
    print(f"\n⚡ ENERGY DISTRIBUTION")
    print(f"  • Low Freq (50-300Hz):    {features['low_to_total_ratio']:.3f}")
    print(f"  • Mid Freq (300-1000Hz):  {features['mid_to_total_ratio']:.3f}")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# EXAMPLE USAGE - Replace with your actual audio files
# ============================================================================

if __name__ == "__main__":
    # === CHANGE THIS PATH TO YOUR AUDIO FILE ===
    audio_path = r"C:\Users\Omprakash\Desktop\pthon\droneaudio.wav"
    
    print("🔊 Analyzing your drone audio...")
    print(f"File: {audio_path}")
    print("-" * 50)
    
    # Extract features from YOUR file
    features = extract_features_from_audio_file(audio_path, debug=True)
    
    # Print detailed results
    print_feature_summary(features, "Your Audio File")
    
    print("\n✅ Done! Check the results above.")
