import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from audio_processing import extract_features, SAMPLE_RATE  # Import from your file

def load_and_extract(folder, label, sr=SAMPLE_RATE):
    features = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            y, _ = librosa.load(os.path.join(folder, file), sr=sr)
            feats = extract_features(y, sr=sr)
            features.append(feats)
            labels.append(label)
    return features, labels

# Load data (drone = 1, non-drone = 0)
drone_feats, drone_labels = load_and_extract('data/audio_train/drone', 1)
noise_feats, noise_labels = load_and_extract('data/audio_train/noise', 0)
heli_feats, heli_labels = load_and_extract('data/audio_train/helicopter', 0)  # Treat helicopter as noise

# Combine
X = np.vstack(drone_feats + noise_feats + heli_feats)
y = np.array(drone_labels + noise_labels + heli_labels)

# Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM
clf = SVC(kernel='rbf', probability=True, C=1.0, class_weight='balanced')  # Balanced for uneven classes
clf.fit(X_train, y_train)

# Evaluate
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]  # Drone probs
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.2f}")
print(classification_report(y_test, preds))  # Per-class metrics

# Save
joblib.dump(clf, 'models/drone_audio_clf.pkl')
print("Model saved — ready for use in audio_processing.py!")