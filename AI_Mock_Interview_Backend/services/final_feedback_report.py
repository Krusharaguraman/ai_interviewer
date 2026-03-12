import json
import os
from collections import defaultdict, Counter

# =========================
# EMOTION SCORING SYSTEM
# =========================
emotion_scores = {
    "happy": 90,
    "confident": 90,
    "neutral": 75,
    "surprise": 70,
    "sad": 50,
    "fear": 50,
    "angry": 40,
    "disgust": 40,
    "no_face": 60
}

# =========================
# MAIN FUNCTION
# =========================
def generate_verbose_report():
    """
    This function reads the candidate's interview results from a JSON file,
    performs comprehensive analysis including content scoring, face and audio emotion scoring,
    level-wise performance, weak concept identification, answer quality assessment,
    interview readiness evaluation, and generates a long, detailed, paragraph-style feedback report
    suitable for professional review or documentation purposes.
    """
    
    # ------------------------
    # Robust JSON file path
    # ------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    JSON_PATH = os.path.join(BASE_DIR, "interview_results.json")
    if not os.path.exists(JSON_PATH):
        JSON_PATH = os.path.join(os.path.dirname(BASE_DIR), "output", "interview_results.json")
    if not os.path.exists(JSON_PATH):
        print(f"❌ interview_results.json not found at {JSON_PATH}")
        return
    
    # ------------------------
    # Load JSON data
    # ------------------------
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    if not data:
        print("❌ No interview data found.")
        return
    
    # ------------------------
    # Initialize scoring collections
    # ------------------------
    content_scores = []
    face_scores = []
    audio_scores = []
    level_correct = defaultdict(int)
    level_total = defaultdict(int)
    weak_keywords = []

    # ------------------------
    # Process each question-answer entry
    # ------------------------
    for item in data:
        # Content scoring
        content_score = item.get("relevance_score", 0)
        content_scores.append(content_score)
        
        # Face and audio emotion scoring
        face = emotion_scores.get(item.get("dominant_face_emotion", "no_face"), 60)
        audio = emotion_scores.get(item.get("audio_emotion", "neutral"), 60)
        face_scores.append(face)
        audio_scores.append(audio)
        
        # Level-wise correct answers
        level = item.get("level", "unknown")
        level_total[level] += 1
        if content_score >= 80:
            level_correct[level] += 1
        
        # Weak concept detection
        weak_keywords.extend(item.get("missing_keywords", []))
    
    concept_counter = Counter(weak_keywords)
    
    # ------------------------
    # Calculate averages and final scores
    # ------------------------
    avg_content = sum(content_scores) / len(content_scores)
    avg_face = sum(face_scores) / len(face_scores)
    avg_audio = sum(audio_scores) / len(audio_scores)
    final_score = 0.6 * avg_content + 0.25 * avg_face + 0.15 * avg_audio
    readiness_score = 0.7 * avg_content + 0.3 * avg_audio
    
    # ------------------------
    # Determine performance level
    # ------------------------
    if final_score >= 85:
        performance = "Excellent"
        feedback_message = ("The candidate has demonstrated an exceptional level of proficiency and confidence "
                            "throughout the interview process. Their technical knowledge is robust and explanations "
                            "are clear, coherent, and logically structured. Non-verbal communication such as facial "
                            "expressions and vocal tone further reinforce their confident presentation.")
    elif final_score >= 70:
        performance = "Good"
        feedback_message = ("The candidate performed well overall, with a good grasp of the technical concepts "
                            "required. While explanations are generally clear, there are some minor areas that "
                            "could benefit from further clarification. Emotional expressions indicate a moderate "
                            "level of confidence, but occasional hesitation was noted.")
    elif final_score >= 50:
        performance = "Average"
        feedback_message = ("The candidate's performance is acceptable but shows room for improvement. Some answers "
                            "lack technical depth, and clarity of explanation is inconsistent. Non-verbal cues suggest "
                            "moderate engagement but occasional signs of uncertainty. Focused preparation on key topics "
                            "is recommended to enhance both knowledge and confidence.")
    else:
        performance = "Needs Improvement"
        feedback_message = ("The candidate requires substantial improvement in both technical understanding and "
                            "communication. Many answers lacked depth and important concepts were missed. Facial "
                            "expressions and vocal tone indicate low confidence. A structured plan to review core "
                            "concepts and practice mock interviews is strongly recommended.")
    
    # ------------------------
    # Answer quality assessment
    # ------------------------
    strong_answers = sum(1 for s in content_scores if s >= 80)
    moderate_answers = sum(1 for s in content_scores if 50 <= s < 80)
    weak_answers = sum(1 for s in content_scores if s < 50)
    
    # ------------------------
    # Begin verbose report output
    # ------------------------
    print("\n======================================================")
    print("               FINAL INTERVIEW REPORT")
    print("======================================================\n")
    
    print(f"Weighted Final Score: {final_score:.2f} ({performance})")
    print(f"Average Content Score: {avg_content:.2f}")
    print(f"Average Face Emotion Score: {avg_face:.2f}")
    print(f"Average Audio Emotion Score: {avg_audio:.2f}")
    print(f"Overall Interview Readiness Score: {readiness_score:.2f}%\n")
    
    print("------------------------------------------------------")
    print("                 DETAILED PERFORMANCE FEEDBACK")
    print("------------------------------------------------------\n")
    print(feedback_message + "\n")
    
    print("------------------------------------------------------")
    print("               LEVEL-WISE PERFORMANCE ANALYSIS")
    print("------------------------------------------------------\n")
    for lvl in level_total:
        print(f"Level '{lvl}': Correct Answers {level_correct[lvl]} out of {level_total[lvl]}")
    print()
    
    print("------------------------------------------------------")
    print("                  ANSWER QUALITY SUMMARY")
    print("------------------------------------------------------\n")
    print(f"Number of Strong Answers   : {strong_answers}")
    print(f"Number of Moderate Answers : {moderate_answers}")
    print(f"Number of Weak Answers     : {weak_answers}")
    if weak_answers > 0:
        print("\nObservation: Some answers lacked technical depth and key concepts were missed. "
              "Improvement in both technical understanding and articulation is recommended.\n")
    
    print("------------------------------------------------------")
    print("                     SKILL GAP ANALYSIS")
    print("------------------------------------------------------\n")
    if concept_counter:
        for concept, count in concept_counter.most_common(5):
            print(f"Concept '{concept}' was missed or insufficiently explained {count} times. "
                  "Targeted revision of this concept is advised to enhance interview readiness.")
    else:
        print("No significant skill gaps detected. Candidate has demonstrated proficiency across assessed topics.\n")
    
    print("------------------------------------------------------")
    print("                 PERSONALIZED IMPROVEMENT PLAN")
    print("------------------------------------------------------\n")
    suggestions = []
    if weak_answers > 0: suggestions.append("Practice explaining answers with more technical depth, clarity, and confidence.")
    if avg_audio < 70: suggestions.append("Work on vocal clarity, modulation, and emotional expressiveness to convey confidence.")
    if concept_counter: 
        top_concepts = [c for c,_ in concept_counter.most_common(3)]
        suggestions.append(f"Revise key concepts related to: {', '.join(top_concepts)} for stronger technical foundation.")
    suggestions.append("Engage in regular mock interviews to simulate real scenarios and improve communication skills.")
    
    for i, s in enumerate(suggestions, 1):
        print(f"{i}. {s}")
    print("\n======================================================\n")

# =========================
# RUN SCRIPT
# =========================
if __name__ == "__main__":
    generate_verbose_report()