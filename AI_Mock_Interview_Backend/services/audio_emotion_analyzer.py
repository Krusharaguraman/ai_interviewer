import os
import json
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================= BASE PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_emotion_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

TESS_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "TESS")

AUDIO_ANSWERS_DIR = os.path.join(BASE_DIR, "output", "audio_answers")
JSON_PATH = os.path.join(BASE_DIR, "output", "interview_results.json")


# ================= FEATURE EXTRACTION =================
def extract_features(file_path):

    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        )

        return np.mean(mfcc.T, axis=0)

    except Exception as e:
        print("Feature extraction error:", e)
        return None


# ================= LOAD OR TRAIN MODEL =================
def load_or_train_model():

    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):

        print("Loading pre-trained audio emotion model...")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)

        return model, le

    print("Training audio emotion model from TESS dataset...")

    features = []
    labels = []

    for root, dirs, files in os.walk(TESS_DATASET_PATH):

        for file in files:

            if file.endswith(".wav"):

                path = os.path.join(root, file)

                folder = os.path.basename(root)

                emotion = folder.split("_")[-1].lower()

                feature = extract_features(path)

                if feature is not None:

                    features.append(feature)
                    labels.append(emotion)

    if len(features) == 0:

        print("TESS dataset not found!")

        return None, None

    X = np.array(features)
    y = np.array(labels)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("Audio Emotion Model Accuracy:", round(acc * 100, 2), "%")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    return model, le


# ================= AUDIO EMOTION ANALYSIS =================
def analyze_audio_emotions():

    print("\n🎧 Running Audio Emotion Analysis...\n")

    model, le = load_or_train_model()

    if model is None:
        print("Audio model not available.")
        return

    if not os.path.exists(JSON_PATH):

        print("interview_results.json not found.")
        return

    with open(JSON_PATH, "r") as f:
        interview_data = json.load(f)

    for item in interview_data:

        qid = item["question_id"]

        audio_file = os.path.join(
            AUDIO_ANSWERS_DIR,
            f"question_{qid}.wav"
        )

        if not os.path.exists(audio_file):

            item["audio_emotion"] = "audio_not_found"

            print(f"Q{qid}: audio file missing")

            continue

        feature = extract_features(audio_file)

        if feature is None:

            item["audio_emotion"] = "error"

            print(f"Q{qid}: feature extraction failed")

            continue

        feature = feature.reshape(1, -1)

        try:

            prediction = model.predict(feature)

            emotion = le.inverse_transform(prediction)[0]

        except:

            emotion = "unknown"

        item["audio_emotion"] = emotion

        print(f"Q{qid} Audio Emotion → {emotion}")

    with open(JSON_PATH, "w") as f:

        json.dump(interview_data, f, indent=4)

    print("\n✅ Audio emotion analysis completed\n")


# ================= RUN DIRECT =================
if __name__ == "__main__":

    analyze_audio_emotions()