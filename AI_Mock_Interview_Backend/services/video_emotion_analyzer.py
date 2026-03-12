import json
import os

# deepface is optional; if not installed, analysis will be skipped
try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

# ================= BASE DIR =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "output", "interview_results.json")
VIDEO_FOLDER = os.path.join(BASE_DIR, "output", "video_answers")


# ================= FUNCTION =================
def analyze_video_emotions():

    if DeepFace is None:
        print("⚠️ deepface library is not installed; video emotion analysis cannot run.")
        return

    if not os.path.exists(JSON_PATH):
        print("interview_results.json not found. Run interview first.")
        return

    # Load JSON
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # Function to get dominant emotion from video
    def get_dominant_emotion(video_path):
        try:
            result = DeepFace.analyze(video_path, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list) and len(result) > 0:
                dominant = result[0].get("dominant_emotion", "no_face")
                return dominant.lower() if dominant else "no_face"

            elif isinstance(result, dict):
                return result.get("dominant_emotion", "no_face").lower()

            return "no_face"

        except Exception as e:
            print(f"Error analyzing {video_path}: {e}")
            return "no_face"

    # Update emotions
    updated_count = 0

    for item in data:

        question_id = item["question_id"]

        if not os.path.exists(VIDEO_FOLDER):
            print("Video folder not found.")
            return

        matching_videos = [
            f for f in os.listdir(VIDEO_FOLDER)
            if f.startswith(f"question_{question_id}") and f.endswith(".mp4")
        ]

        dominant_emotion = "no_face"

        for video_file in matching_videos:
            video_path = os.path.join(VIDEO_FOLDER, video_file)

            emotion = get_dominant_emotion(video_path)

            if emotion != "no_face":
                dominant_emotion = emotion
                break

        # write back using the standardized key
        item["dominant_face_emotion"] = dominant_emotion
        # support legacy key for backwards compatibility
        item["dominant_emotion"] = dominant_emotion

        if dominant_emotion != "no_face":
            updated_count += 1

        print(f"Q{question_id}: {dominant_emotion}")

    # Save updated JSON
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\n✅ Video emotions updated! {updated_count}/{len(data)} questions analyzed.")


# ================= RUN DIRECTLY =================
if __name__ == "__main__":
    analyze_video_emotions()