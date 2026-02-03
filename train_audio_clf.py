# test_audio_rule_based.py
# -------------------------------------------------
# Chunk-wise testing for rule-based drone detector
# -------------------------------------------------

import librosa
from audio_rule_based import detect_drone_audio, SAMPLE_RATE

audio_file = r"C:\Users\Omprakash\Desktop\pthon\nature.wav"

y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)

CHUNK_SEC = 2.0
hop = int(CHUNK_SEC * sr)

print("\n=== AUDIO DETECTION RESULTS ===")

for i in range(0, len(y) - hop, hop):
    chunk = y[i:i + hop]
    result = detect_drone_audio(chunk, sr, debug=False)

    print(f"\nt = {i/sr:.2f}s")
    for k, v in result.items():
        print(f"{k}: {v}")