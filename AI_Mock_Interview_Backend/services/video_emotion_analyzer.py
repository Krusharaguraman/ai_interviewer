import json
import os
from deepface import DeepFace  # pip install deepface

# Paths
json_path = r"D:\finalyearprojecttesting\mypart\AI_Mock_Interview_Backend\output\interview_results.json"
video_folder = r"D:\finalyearprojecttesting\mypart\AI_Mock_Interview_Backend\output\video_answers"

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Function to get dominant emotion from video
def get_dominant_emotion(video_path):
    try:
        # Analyze video for emotions
        result = DeepFace.analyze(video_path, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list) and len(result) > 0:
            dominant = result[0].get("dominant_emotion", "no_face")
            return dominant if dominant else "no_face"
        elif isinstance(result, dict):
            return result.get("dominant_emotion", "no_face")
        else:
            return "no_face"
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}")
        return "no_face"

# Loop through each question
for item in data:
    question_id = item["question_id"]
    
    # Handle multiple videos per question: question_1.mp4, question_1_1.mp4, question_1_2.mp4 ...
    matching_videos = [f for f in os.listdir(video_folder) if f.startswith(f"question_{question_id}")]
    
    dominant_emotion = "no_face"
    for video_file in matching_videos:
        video_path = os.path.join(video_folder, video_file)
        dominant_emotion = get_dominant_emotion(video_path)
        if dominant_emotion != "no_face":
            break  # take the first valid emotion

    item["dominant_emotion"] = dominant_emotion

# Save updated JSON
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

print("Dominant emotions updated successfully!")