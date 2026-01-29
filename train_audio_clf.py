import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from audio_processing import extract_features, SAMPLE_RATE  # Your file

def load_and_extract(folder, label, sr=SAMPLE_RATE):
    features = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            path = os.path.join(folder, file)
            y, _ = librosa.load(path, sr=sr)
            feats = extract_features(y, sr=sr)
            features.append(feats)
            labels.append(label)
    return features, labels

# Load folders
drone_feats, drone_labels = load_and_extract(r'C:\Users\Omprakash\Desktop\pthon\data\drone', 1)  # Drone = 1
noise_feats, noise_labels = load_and_extract(r'C:\Users\Omprakash\Desktop\pthon\data\noise', 0)  # Noise = 0

# Combine
X = np.vstack(drone_feats + noise_feats)
y = np.array(drone_labels + noise_labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM (RBF kernel, balanced for imbalance)
clf = SVC(kernel='rbf', probability=True, C=1.0, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.2f}")
print(classification_report(y_test, preds, target_names=['Noise', 'Drone']))

# Save
joblib.dump(clf, 'models/drone_audio_clf.pkl')
print("Model saved — ready for runtime!")