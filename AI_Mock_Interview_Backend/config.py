import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TESS_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "TESS")
SKILL_CSV_PATH = os.path.join(BASE_DIR, "dataset", "technical_skills.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_emotion_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "audio")