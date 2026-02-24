import os
import json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================= FEATURE EXTRACTION =================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Error extracting:", file_path, e)
        return None


# ================= DATASET PATH =================
dataset_path = r"D:\finalyearprojecttesting\mypart\AI_Mock_Interview_Backend\dataset\TESS"

print("Dataset Path Exists:", os.path.exists(dataset_path))

features = []
labels = []

# ================= LOAD DATASET =================
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            # Extract emotion from folder name
            folder_name = os.path.basename(root)
            emotion = folder_name.split("_")[-1].lower()

            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(emotion)

print("Total samples found:", len(features))

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ================= TRAIN RANDOM FOREST =================
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ================= EVALUATE MODEL =================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")


# =====================================================
#        PREDICT INTERVIEW AUDIO & UPDATE JSON
# =====================================================

audio_folder = r"D:\finalyearprojecttesting\mypart\AI_Mock_Interview_Backend\output\audio_answers"
json_path = r"D:\finalyearprojecttesting\mypart\AI_Mock_Interview_Backend\output\interview_results.json"

# Load JSON
with open(json_path, "r") as f:
    interview_data = json.load(f)

# Loop through each question
for item in interview_data:
    question_id = item["question_id"]

    audio_file = os.path.join(audio_folder, f"question_{question_id}.wav")

    if os.path.exists(audio_file):
        feature = extract_features(audio_file)

        if feature is not None:
            feature = feature.reshape(1, -1)
            prediction = model.predict(feature)
            emotion = le.inverse_transform(prediction)[0]

            # Update only audio_emotion field
            item["audio_emotion"] = emotion
            print(f"Updated Question {question_id} → {emotion}")
        else:
            item["audio_emotion"] = "error"
    else:
        item["audio_emotion"] = "audio_not_found"

# Save updated JSON
with open(json_path, "w") as f:
    json.dump(interview_data, f, indent=4)

print("\n✅ JSON updated successfully with audio emotions!")