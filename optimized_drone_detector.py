"""
Optimized Drone Detection Rules
Based on YOUR actual test results from forest environment

Results from your tests:
- Drone Only: 7 peaks, flatness 0.1158, harmonic_power 0.461
- Drone + Forest: 1 peak, flatness 0.1035, harmonic_power 0.003
- Forest Only: 0 peaks, flatness 0.0456, harmonic_power 0.000
"""

def detect_drone_optimized_for_forest(features):
    """
    Optimized for forest environment with wind noise
    
    Key discriminators for YOUR environment:
    1. Number of harmonic peaks (most important)
    2. Harmonic power ratio (critical for weak drones)
    3. Spectral flatness (helps distinguish drone from wind)
    4. Low vs Mid frequency energy ratio
    
    Returns:
        is_drone (bool): True if drone detected
        confidence (float): 0-1 confidence score
        details (dict): Breakdown of decision
    """
    
    score = 0
    max_score = 10
    details = {}
    
    # ===================================================================
    # RULE 1: Number of Harmonic Peaks (WEIGHT: 4.0) - MOST IMPORTANT
    # ===================================================================
    # Your results:
    #   Drone: 7 peaks
    #   Weak Drone: 1 peak
    #   Forest: 0 peaks
    
    num_peaks = features['num_harmonic_peaks']
    
    if num_peaks >= 5:
        score += 4.0
        details['peaks_score'] = 4.0
        details['peaks_reason'] = f"Strong drone signature ({num_peaks} peaks)"
    elif num_peaks >= 3:
        score += 3.0
        details['peaks_score'] = 3.0
        details['peaks_reason'] = f"Moderate drone signature ({num_peaks} peaks)"
    elif num_peaks >= 1:
        score += 1.5
        details['peaks_score'] = 1.5
        details['peaks_reason'] = f"Weak drone signature ({num_peaks} peak)"
    else:
        score += 0
        details['peaks_score'] = 0
        details['peaks_reason'] = "No harmonic peaks detected"
    
    # ===================================================================
    # RULE 2: Harmonic Power Ratio (WEIGHT: 2.5) - CRITICAL FOR WEAK DRONES
    # ===================================================================
    # Your results:
    #   Drone: 0.461 (high)
    #   Weak Drone: 0.003 (very low but > 0)
    #   Forest: 0.000 (zero)
    
    harmonic_power = features['harmonic_power_ratio']
    
    if harmonic_power > 0.3:
        score += 2.5
        details['harmonic_power_score'] = 2.5
        details['harmonic_power_reason'] = f"Strong harmonic content ({harmonic_power:.3f})"
    elif harmonic_power > 0.1:
        score += 1.5
        details['harmonic_power_score'] = 1.5
        details['harmonic_power_reason'] = f"Moderate harmonic content ({harmonic_power:.3f})"
    elif harmonic_power > 0.001:
        score += 0.5
        details['harmonic_power_score'] = 0.5
        details['harmonic_power_reason'] = f"Weak harmonic content ({harmonic_power:.3f})"
    else:
        score += 0
        details['harmonic_power_score'] = 0
        details['harmonic_power_reason'] = "No harmonic content"
    
    # ===================================================================
    # RULE 3: Spectral Flatness (WEIGHT: 2.0) - TONAL VS NOISE
    # ===================================================================
    # Your results:
    #   Drone: 0.1158 (tonal)
    #   Weak Drone: 0.1035 (still tonal)
    #   Forest: 0.0456 (unexpectedly low - forest wind has some tonality)
    
    flatness = features['spectral_flatness_mean']
    
    if flatness > 0.15:
        # Very noise-like - probably not drone
        score -= 1.0
        details['flatness_score'] = -1.0
        details['flatness_reason'] = f"Too noise-like ({flatness:.4f})"
    elif flatness > 0.08:
        # Somewhat tonal - could be weak drone
        score += 1.0
        details['flatness_score'] = 1.0
        details['flatness_reason'] = f"Moderate tonality ({flatness:.4f})"
    else:
        # Very tonal OR forest wind (0.04-0.08 range)
        # Only add points if we have peaks
        if num_peaks > 0:
            score += 2.0
            details['flatness_score'] = 2.0
            details['flatness_reason'] = f"High tonality with peaks ({flatness:.4f})"
        else:
            score += 0
            details['flatness_score'] = 0
            details['flatness_reason'] = f"Tonal but no peaks - forest wind? ({flatness:.4f})"
    
    # ===================================================================
    # RULE 4: Energy Distribution (WEIGHT: 1.5) - FOREST WIND FILTER
    # ===================================================================
    # Your results:
    #   Drone: Low=1.084, Mid=4.537 (mid > low)
    #   Weak Drone: Low=23.827, Mid=3.110 (low >> mid - forest dominates!)
    #   Forest: Low=28.334, Mid=3.227 (low >>> mid)
    
    low_energy = features['low_to_total_ratio']
    mid_energy = features['mid_to_total_ratio']
    
    # Forest wind has MUCH higher low-frequency energy
    if low_energy > 10 and mid_energy < 5:
        # Dominated by low-frequency noise (forest wind)
        score -= 0.5
        details['energy_score'] = -0.5
        details['energy_reason'] = f"Forest wind signature (Low={low_energy:.1f}, Mid={mid_energy:.1f})"
    elif mid_energy > low_energy:
        # Drone signature (more mid-frequency)
        score += 1.5
        details['energy_score'] = 1.5
        details['energy_reason'] = f"Drone frequency profile (Low={low_energy:.1f}, Mid={mid_energy:.1f})"
    else:
        score += 0
        details['energy_score'] = 0
        details['energy_reason'] = f"Neutral energy distribution"
    
    # ===================================================================
    # DECISION THRESHOLDS
    # ===================================================================
    
    # Conservative threshold (fewer false positives)
    threshold_conservative = 5.0
    
    # Balanced threshold (recommended)
    threshold_balanced = 3.5
    
    # Sensitive threshold (catch weak drones, more false positives)
    threshold_sensitive = 2.0
    
    # Use balanced threshold by default
    threshold = threshold_balanced
    
    is_drone = score >= threshold
    confidence = min(max(score / max_score, 0), 1.0)
    
    # Add details
    details['total_score'] = score
    details['max_score'] = max_score
    details['threshold'] = threshold
    details['all_thresholds'] = {
        'conservative': threshold_conservative,
        'balanced': threshold_balanced,
        'sensitive': threshold_sensitive
    }
    
    return is_drone, confidence, details


def print_detection_result(features, audio_name=""):
    """Print detailed detection results"""
    
    is_drone, confidence, details = detect_drone_optimized_for_forest(features)
    
    print("\n" + "="*70)
    print(f"  DETECTION RESULT: {audio_name}".center(70))
    print("="*70)
    
    # Main result
    if is_drone:
        print(f"\n🎯 DRONE DETECTED")
        print(f"   Confidence: {confidence:.1%}")
    else:
        print(f"\n❌ NO DRONE DETECTED")
        print(f"   Confidence: {(1-confidence):.1%} it's NOT a drone")
    
    # Score breakdown
    print(f"\n📊 Score Breakdown:")
    print(f"   Total Score: {details['total_score']:.1f} / {details['max_score']}")
    print(f"   Threshold: {details['threshold']:.1f} (balanced)")
    
    # Individual rules
    print(f"\n📋 Rule Contributions:")
    print(f"   1. Harmonic Peaks:   +{details['peaks_score']:.1f}")
    print(f"      → {details['peaks_reason']}")
    
    print(f"   2. Harmonic Power:   +{details['harmonic_power_score']:.1f}")
    print(f"      → {details['harmonic_power_reason']}")
    
    print(f"   3. Spectral Flatness: {details['flatness_score']:+.1f}")
    print(f"      → {details['flatness_reason']}")
    
    print(f"   4. Energy Distribution: {details['energy_score']:+.1f}")
    print(f"      → {details['energy_reason']}")
    
    # Alternative thresholds
    print(f"\n🎚️  Alternative Thresholds:")
    for name, thresh in details['all_thresholds'].items():
        result = "✓ DRONE" if details['total_score'] >= thresh else "✗ NOT DRONE"
        print(f"   {name.capitalize():15s} (≥{thresh:.1f}): {result}")
    
    print("="*70 + "\n")
    
    return is_drone, confidence


# ===================================================================
# EXPECTED RESULTS ON YOUR TEST FILES
# ===================================================================

def test_on_your_data():
    """
    Expected results based on your actual feature extraction
    """
    
    print("\n" + "="*70)
    print("  EXPECTED RESULTS ON YOUR TEST FILES".center(70))
    print("="*70)
    
    # File 1: Drone Only
    features_drone = {
        'num_harmonic_peaks': 7,
        'harmonic_power_ratio': 0.461,
        'spectral_flatness_mean': 0.1158,
        'low_to_total_ratio': 1.084,
        'mid_to_total_ratio': 4.537
    }
    
    print("\n1. droneaudio.wav (Drone Only)")
    print("-" * 70)
    is_drone, conf, details = detect_drone_optimized_for_forest(features_drone)
    print(f"Expected: DRONE DETECTED ✓")
    print(f"Score: {details['total_score']:.1f}/10")
    print(f"Result: {'DRONE' if is_drone else 'NOT DRONE'} (confidence: {conf:.1%})")
    
    # File 2: Weak Drone + Forest
    features_weak = {
        'num_harmonic_peaks': 1,
        'harmonic_power_ratio': 0.003,
        'spectral_flatness_mean': 0.1035,
        'low_to_total_ratio': 23.827,
        'mid_to_total_ratio': 3.110
    }
    
    print("\n2. realtest.wav (Far Drone + Forest Noise)")
    print("-" * 70)
    is_drone, conf, details = detect_drone_optimized_for_forest(features_weak)
    print(f"Expected: BORDERLINE (very weak drone)")
    print(f"Score: {details['total_score']:.1f}/10")
    print(f"Result: {'DRONE' if is_drone else 'NOT DRONE'} (confidence: {conf:.1%})")
    print(f"Note: With 'sensitive' threshold (≥2.0): {'DETECTED' if details['total_score'] >= 2.0 else 'NOT DETECTED'}")
    
    # File 3: Forest Only
    features_forest = {
        'num_harmonic_peaks': 0,
        'harmonic_power_ratio': 0.000,
        'spectral_flatness_mean': 0.0456,
        'low_to_total_ratio': 28.334,
        'mid_to_total_ratio': 3.227
    }
    
    print("\n3. nature.wav (Forest Noise Only)")
    print("-" * 70)
    is_drone, conf, details = detect_drone_optimized_for_forest(features_forest)
    print(f"Expected: NO DRONE ✓")
    print(f"Score: {details['total_score']:.1f}/10")
    print(f"Result: {'DRONE' if is_drone else 'NOT DRONE'} (confidence: {(1-conf):.1%} it's noise)")
    
    print("\n" + "="*70)


# ===================================================================
# RECOMMENDATIONS
# ===================================================================

RECOMMENDATIONS = """
================================================================================
                            RECOMMENDATIONS
================================================================================

Based on your test results, here's what I recommend:

1. THRESHOLD SELECTION:
   -------------------
   For your forest environment:
   
   • BALANCED (3.5) - RECOMMENDED
     ✓ Detects strong drones (File 1): YES
     ✓ Detects weak drones (File 2): MAYBE (score 2.0)
     ✓ Rejects forest noise (File 3): YES
     
   • SENSITIVE (2.0) - If missing drones is worse than false alarms
     ✓ Detects weak drones (File 2): YES
     ⚠️ May have more false positives in very windy conditions
     
   • CONSERVATIVE (5.0) - If false alarms are expensive
     ✓ Only detects strong/clear drones (File 1)
     ✗ Will miss weak/far drones (File 2)

2. FILE 2 CHALLENGE (Far Drone + Forest):
   ---------------------------------------
   This file only detected 1 harmonic peak with very weak power (0.003).
   The drone is VERY far or quiet compared to forest noise.
   
   Options:
   a) Use SENSITIVE threshold (2.0) to catch it
   b) Accept that very far drones may be missed
   c) Use multiple sensors closer to expected drone locations
   d) Apply noise reduction pre-processing

3. DEPLOYMENT STRATEGY:
   --------------------
   • Start with BALANCED threshold (3.5)
   • Deploy and monitor false positive rate
   • If missing too many drones: lower to 2.5-3.0
   • If too many false alarms: raise to 4.0-4.5
   • Consider multi-window fusion (require detection in 3/5 consecutive windows)

4. MULTI-WINDOW FUSION (Recommended):
   -----------------------------------
   Instead of detecting on single window, require detection in multiple
   consecutive windows to reduce false positives:
   
   if detected_in_3_out_of_5_windows:
       trigger_alert()
   
   This is especially useful for File 2 (weak drone) scenarios.

5. NOISE REDUCTION OPTION:
   -----------------------
   For very noisy environments, consider pre-processing with:
   • Spectral subtraction (estimate forest noise profile)
   • Wiener filtering
   • Band-pass filter (focus on 100-2000 Hz drone range)

================================================================================
"""

if __name__ == "__main__":
    test_on_your_data()
    print(RECOMMENDATIONS)
