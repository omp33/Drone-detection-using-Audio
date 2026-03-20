"""
ROBUST Feature Extraction for Drone Detection
Handles edge cases, validates data, provides detailed logging

Key improvements:
- Error handling for corrupted audio
- Validation of extracted features
- Detailed logging for debugging
- Handles silence, clipping, short clips
- Feature normalization
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings('ignore')


class RobustDroneFeatureExtractor:
    """
    Production-grade feature extractor with error handling
    """
    
    def __init__(self, sr=44100, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_names = []  # Will be populated after first extraction
    
    def validate_audio(self, y, filepath=""):
        """
        Validate audio quality
        Returns: (is_valid, error_message)
        """
        # Check for silence
        if np.max(np.abs(y)) < 0.001:
            return False, f"Audio is silent (max amplitude: {np.max(np.abs(y)):.6f})"
        
        # Check for clipping
        clipping_ratio = np.sum(np.abs(y) > 0.99) / len(y)
        if clipping_ratio > 0.01:  # More than 1% clipped
            return False, f"Audio is clipped ({clipping_ratio*100:.1f}% of samples)"
        
        # Check length
        if len(y) < self.sr * 0.5:  # Less than 0.5 seconds
            return False, f"Audio too short ({len(y)/self.sr:.2f} seconds)"
        
        # Check for NaN or Inf
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return False, "Audio contains NaN or Inf values"
        
        return True, "OK"
    
    def extract_features_safe(self, y, sr, filepath=""):
        """
        Extract features with error handling
        Returns: (features_dict, success, error_message)
        """
        try:
            # Validate audio
            is_valid, error_msg = self.validate_audio(y, filepath)
            if not is_valid:
                return {}, False, error_msg
            
            # Normalize audio
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # Extract features
            features = self._extract_all_features(y, sr)
            
            # Validate features (check for NaN/Inf)
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    return {}, False, f"Invalid feature value: {key} = {value}"
            
            return features, True, "OK"
        
        except Exception as e:
            return {}, False, f"Exception during extraction: {str(e)}"
    
    def _extract_all_features(self, y, sr):
        """
        Extract all features (internal method)
        """
        features = {}
        
        # Compute spectrogram
        f, t, S = scipy_signal.stft(
            y, fs=sr, nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )
        S = np.abs(S)
        S_db = 20 * np.log10(S + 1e-10)
        
        # Extract feature groups
        features.update(self._harmonic_features(S, f))
        features.update(self._spectral_features(S, f, y, sr))
        features.update(self._temporal_features(y, sr))
        features.update(self._pattern_features(S_db))
        features.update(self._energy_features(S, f))
        
        return features
    
    def _harmonic_features(self, S, f):
        """Extract harmonic features"""
        features = {}
        
        # Average spectrum
        avg_spectrum = np.mean(S, axis=1)
        
        # Focus on drone range
        drone_mask = (f >= 50) & (f <= 2500)
        drone_spectrum = avg_spectrum[drone_mask]
        drone_freqs = f[drone_mask]
        
        if len(drone_spectrum) == 0:
            # Fallback values
            return {
                'num_harmonic_peaks': 0,
                'fundamental_freq': 0.0,
                'harmonic_regularity': 0.0,
                'harmonic_power_ratio': 0.0,
                'harmonic_spacing_std': 0.0,
                'peak_prominence_mean': 0.0
            }
        
        # Adaptive peak detection
        max_val = np.max(drone_spectrum)
        mean_val = np.mean(drone_spectrum)
        
        # Noise floor estimation
        sorted_vals = np.sort(drone_spectrum)
        noise_floor = np.mean(sorted_vals[:len(sorted_vals)//4])
        
        # Multiple prominence thresholds
        thresholds = [max_val * 0.02, max_val * 0.05, max_val * 0.10]
        
        best_peaks = None
        best_properties = None
        
        for thresh in thresholds:
            peaks, properties = find_peaks(
                drone_spectrum,
                prominence=thresh,
                distance=3,
                height=noise_floor + mean_val * 0.1
            )
            
            if 3 <= len(peaks) <= 15:
                best_peaks = peaks
                best_properties = properties
                break
        
        # If no good detection, use most sensitive
        if best_peaks is None:
            peaks, properties = find_peaks(
                drone_spectrum,
                prominence=thresholds[0],
                distance=3
            )
            best_peaks = peaks
            best_properties = properties
        
        features['num_harmonic_peaks'] = len(best_peaks)
        
        if len(best_peaks) >= 1:
            peak_freqs = drone_freqs[best_peaks]
            features['fundamental_freq'] = float(peak_freqs[0])
            
            if len(best_peaks) >= 2:
                spacing = np.diff(peak_freqs)
                features['harmonic_spacing_std'] = float(np.std(spacing))
                features['harmonic_regularity'] = float(1.0 / (1.0 + np.std(spacing) / 50.0))
            else:
                features['harmonic_spacing_std'] = 0.0
                features['harmonic_regularity'] = 0.0
            
            peak_power = np.sum(drone_spectrum[best_peaks]**2)
            total_power = np.sum(drone_spectrum**2)
            features['harmonic_power_ratio'] = float(peak_power / (total_power + 1e-10))
            
            features['peak_prominence_mean'] = float(np.mean(best_properties['prominences']))
        else:
            features['fundamental_freq'] = 0.0
            features['harmonic_spacing_std'] = 0.0
            features['harmonic_regularity'] = 0.0
            features['harmonic_power_ratio'] = 0.0
            features['peak_prominence_mean'] = 0.0
        
        return features
    
    def _spectral_features(self, S, f, y, sr):
        """Extract spectral features"""
        features = {}
        
        avg_spectrum = np.mean(S, axis=1)
        
        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(avg_spectrum + 1e-10)))
        arithmetic_mean = np.mean(avg_spectrum)
        features['spectral_flatness_mean'] = float(geometric_mean / (arithmetic_mean + 1e-10))
        
        # Spectral centroid
        features['spectral_centroid_mean'] = float(np.sum(f * avg_spectrum) / (np.sum(avg_spectrum) + 1e-10))
        
        # Spectral spread
        centroid = features['spectral_centroid_mean']
        features['spectral_spread'] = float(
            np.sqrt(np.sum(((f - centroid)**2) * avg_spectrum) / (np.sum(avg_spectrum) + 1e-10))
        )
        
        # Spectral contrast
        bands = [(50, 300), (300, 1000), (1000, 2500)]
        contrasts = []
        for low, high in bands:
            band_mask = (f >= low) & (f < high)
            if np.any(band_mask):
                band_spec = avg_spectrum[band_mask]
                contrast = np.percentile(band_spec, 90) - np.percentile(band_spec, 10)
                contrasts.append(contrast)
        features['spectral_contrast_mean'] = float(np.mean(contrasts)) if contrasts else 0.0
        
        # Spectral rolloff
        cumsum = np.cumsum(avg_spectrum)
        rolloff_threshold = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        features['spectral_rolloff'] = float(f[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(f[-1])
        
        return features
    
    def _temporal_features(self, y, sr):
        """Extract temporal features"""
        features = {}
        
        # RMS energy (even though it's not discriminative, include for completeness)
        frame_length = 2048
        hop = 512
        rms_frames = []
        for i in range(0, len(y) - frame_length, hop):
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
    
    def _pattern_features(self, S_db):
        """Extract spectrogram pattern features"""
        features = {}
        
        # Horizontal band strength
        freq_variance = np.var(S_db, axis=1)
        features['freq_variance_mean'] = float(np.mean(freq_variance))
        features['horizontal_band_strength'] = float(1.0 / (1.0 + np.mean(freq_variance)))
        
        # Temporal correlation
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
    
    def _energy_features(self, S, f):
        """Extract energy distribution features"""
        features = {}
        
        bands = {
            'low': (50, 300),
            'mid': (300, 1000),
            'high': (1000, 2500),
            'vhigh': (2500, f[-1])
        }
        
        total_energy = np.mean(S)
        
        for band_name, (low, high) in bands.items():
            mask = (f >= low) & (f < high)
            if np.any(mask):
                band_energy = np.mean(S[mask, :])
                features[f'energy_{band_name}_freq'] = float(band_energy)
                features[f'{band_name}_to_total_ratio'] = float(band_energy / (total_energy + 1e-10))
            else:
                features[f'energy_{band_name}_freq'] = 0.0
                features[f'{band_name}_to_total_ratio'] = 0.0
        
        return features
    
    def extract_from_dataset(self, data_dir, output_csv='testfeatures.csv'):
        """
        Extract features from organized dataset
        
        data_dir/
          drone/
            *.wav
          background/
            *.wav
        """
        print("="*80)
        print("ROBUST FEATURE EXTRACTION".center(80))
        print("="*80)
        
        results = []
        errors = []
        
        for class_name, label in [('drone', 1), ('background', 0)]:
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"\n⚠️  Warning: Directory not found: {class_dir}")
                continue
            
            audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            
            print(f"\n📁 Processing {class_name}: {len(audio_files)} files")
            
            for audio_file in tqdm(audio_files, desc=f"  Extracting {class_name}"):
                filepath = os.path.join(class_dir, audio_file)
                
                try:
                    # Load audio
                    y, sr = librosa.load(filepath, sr=self.sr)
                    
                    # Extract features
                    features, success, error_msg = self.extract_features_safe(y, sr, filepath)
                    
                    if success:
                        features['label'] = label
                        features['filename'] = audio_file
                        features['class'] = class_name
                        results.append(features)
                    else:
                        errors.append({
                            'file': audio_file,
                            'class': class_name,
                            'error': error_msg
                        })
                
                except Exception as e:
                    errors.append({
                        'file': audio_file,
                        'class': class_name,
                        'error': f"Load error: {str(e)}"
                    })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save
        df.to_csv(output_csv, index=False)
        
        # Summary
        print(f"\n{'='*80}")
        print("EXTRACTION SUMMARY".center(80))
        print(f"{'='*80}")
        print(f"\n✓ Successfully processed: {len(results)} files")
        print(f"  Drone: {len([r for r in results if r['label'] == 1])}")
        print(f"  Background: {len([r for r in results if r['label'] == 0])}")
        
        if errors:
            print(f"\n✗ Errors: {len(errors)} files")
            for err in errors[:5]:  # Show first 5 errors
                print(f"  {err['file']}: {err['error']}")
            if len(errors) > 5:
                print(f"  ... and {len(errors)-5} more errors")
            
            # Save error log
            with open('extraction_errors.json', 'w') as f:
                json.dump(errors, f, indent=2)
            print(f"\n  Full error log saved to: extraction_errors.json")
        
        print(f"\n✓ Features saved to: {output_csv}")
        print(f"  Total features per sample: {len(df.columns) - 3}")  # Exclude label, filename, class
        
        # Save feature names
        feature_cols = [col for col in df.columns if col not in ['label', 'filename', 'class']]
        self.feature_names = feature_cols
        
        with open('feature_names.json', 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"✓ Feature names saved to: feature_names.json")
        
        print(f"\n{'='*80}\n")
        
        return df, errors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features robustly')
    parser.add_argument('--data_dir', required=True, help='Dataset directory (with drone/ and background/ subdirs)')
    parser.add_argument('--output', default='features.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    extractor = RobustDroneFeatureExtractor()
    df, errors = extractor.extract_from_dataset(args.data_dir, args.output)
    
    print("✅ Feature extraction complete!")