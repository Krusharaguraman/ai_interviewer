import os
import json
import time
import threading
import sounddevice as sd
import soundfile as sf
import cv2
import spacy
from sentence_transformers import SentenceTransformer, util
import ollama
from faster_whisper import WhisperModel
import re

# =========================
# CONFIGURATION
# =========================
BASE_DIR = "D:/finalyearprojecttesting/mypart/AI_Mock_Interview_Backend"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio_answers")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video_answers")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

MAX_TOTAL_QUESTIONS = 20
LEVELS = ["Beginner", "Intermediate", "Advanced"]
AUDIO_DURATION = 30  # seconds for testing
THINKING_TIMER = 0
WHISPER_MODEL_SIZE = "small"  # tiny/small/base

# =========================
# LOAD MODELS
# =========================
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading Whisper model...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, compute_type="float32")

print("Loading NLP model...")
nlp = spacy.load("en_core_web_sm")

# =========================
# LOAD SKILLS
# =========================
def load_skills():
    path = os.path.join(OUTPUT_DIR, "extracted_skills.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data["extracted_skills"]

# =========================
# LLM CACHE
# =========================
qa_cache = {}

# =========================
# SINGLE LLM CALL: QUESTION + REFERENCE ANSWER
# =========================
def generate_question_and_answer(skill, level, context=None):
    key = f"{skill}_{level}_{context if context else ''}"
    if key in qa_cache:
        return qa_cache[key]

    prompt = (
        f"Generate one {level} level oral technical HR interview question for {skill}"
    )
    if context:
        prompt += f" focusing on this concept or keyword: {context}"

    prompt += (
        ". Provide a concise one line reference answer. Return output in strict JSON format only like this:\n"
        "{\n  \"question\": \"Your question here\",\n"
        "  \"answer\": \"Your concise answer here\"\n}"
    )

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    content = response["message"]["content"].strip()

    # Robust JSON extraction
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        json_text = match.group()
        try:
            qa = json.loads(json_text)
            question = qa.get("question", "No question found")
            ref_answer = qa.get("answer", "No answer found")
        except json.JSONDecodeError:
            question = "No question found"
            ref_answer = "No answer found"
    else:
        question = "No question found"
        ref_answer = "No answer found"

    qa_cache[key] = (question, ref_answer)
    return question, ref_answer

# =========================
# NLP KEYWORDS
# =========================
def extract_core_keywords(text):
    doc = nlp(text.lower())
    return set(
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN", "ADJ"] and len(token.text) > 2
    )

def find_missing_keywords(reference, answer):
    ref_keywords = extract_core_keywords(reference)
    ans_keywords = extract_core_keywords(answer)
    return list(ref_keywords - ans_keywords)

# =========================
# COUNTDOWN
# =========================
def countdown(seconds=THINKING_TIMER):
    for i in range(seconds, 0, -1):
        print(f"Recording starts in {i} seconds...", end="\r")
        time.sleep(1)
    print("\nRecording Started!\n")

# =========================
# RECORD AUDIO + VIDEO
# =========================
def record_audio_video(question_id, duration=AUDIO_DURATION, fs=44100):
    audio_path = os.path.join(AUDIO_DIR, f"question_{question_id}.wav")
    video_path = os.path.join(VIDEO_DIR, f"question_{question_id}.mp4")

    def record_audio():
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        sf.write(audio_path, rec, fs)

    def record_video():
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        start_time = time.time()
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
        cap.release()
        out.release()

    t_audio = threading.Thread(target=record_audio)
    t_video = threading.Thread(target=record_video)
    t_audio.start()
    t_video.start()
    t_audio.join()
    t_video.join()

    print("Recording completed.\n")
    return audio_path, video_path

# =========================
# TRANSCRIBE AUDIO
# =========================
def transcribe_audio(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    return "".join(segment.text for segment in segments).strip()

# =========================
# COSINE SIMILARITY
# =========================
def compute_similarity(answer, reference, reference_emb=None):
    if not answer.strip():
        return 0.0
    emb1 = embed_model.encode(answer, convert_to_tensor=True)
    emb2 = reference_emb if reference_emb is not None else embed_model.encode(reference, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score) * 100

# =========================
# INTERVIEW LOOP WITH GAP FOLLOW-UP
# =========================
def start_interview():
    skills = load_skills()
    results = []
    question_id = 1
    total_questions = 0

    print(f"Total Skills Loaded: {len(skills)}\nInterview Started...\n")

    # Skill → Level progression (no cycling back)
    for skill_obj in skills:
        skill = skill_obj["Skill Name"]

        for level in LEVELS:

            if total_questions >= MAX_TOTAL_QUESTIONS:
                break

            # =========================
            # MAIN QUESTION
            # =========================
            question, ref_answer = generate_question_and_answer(skill, level)
            ref_emb = embed_model.encode(ref_answer, convert_to_tensor=True)

            print(f"\n===== Question {question_id} =====")
            print(f"Skill: {skill}, Level: {level}")
            print("Question:", question)
            print("Reference Answer:", ref_answer)

            countdown(1)
            audio_path, video_path = record_audio_video(question_id)
            answer_text = transcribe_audio(audio_path)

            relevance_score = compute_similarity(answer_text, ref_answer, ref_emb)
            missing_keywords = find_missing_keywords(ref_answer, answer_text)

            print(f"\n----- TRANSCRIBED ANSWER -----")
            print(answer_text)
            print("--------------------------------\n")
            print(f"Relevance Score: {relevance_score:.2f}%")
            print(f"Missing Core Concepts: {missing_keywords}\n")

            results.append({
                "question_id": question_id,
                "skill": skill,
                "level": level,
                "answer": answer_text,
                "reference_answer": ref_answer,
                "relevance_score": relevance_score,
                "missing_keywords": missing_keywords,
                "dominant_emotion": "",
                "audio_emotion": ""
            })

            question_id += 1
            total_questions += 1

            # =========================
            # GAP FOLLOW-UP (IF WEAK)
            # =========================
            if (
                relevance_score < 50
                and missing_keywords
                and total_questions < MAX_TOTAL_QUESTIONS
            ):
                follow_context = ", ".join(missing_keywords[:2])

                print(f"Weak concepts detected: {follow_context}. Asking follow-up question...")

                gap_question, gap_ref_answer = generate_question_and_answer(
                    skill, "Follow-up", context=follow_context
                )

                gap_ref_emb = embed_model.encode(gap_ref_answer, convert_to_tensor=True)

                print(f"\n===== Follow-up Question {question_id} =====")
                print("Question:", gap_question)
                print("Reference Answer:", gap_ref_answer)

                countdown(1)
                audio_path, video_path = record_audio_video(question_id)
                gap_answer_text = transcribe_audio(audio_path)

                gap_score = compute_similarity(
                    gap_answer_text, gap_ref_answer, gap_ref_emb
                )

                gap_missing = find_missing_keywords(
                    gap_ref_answer, gap_answer_text
                )

                print(f"\n----- TRANSCRIBED FOLLOW-UP ANSWER -----")
                print(gap_answer_text)
                print("--------------------------------\n")
                print(f"Follow-up Relevance Score: {gap_score:.2f}%")
                print(f"Remaining Missing Concepts: {gap_missing}\n")

                results.append({
                    "question_id": question_id,
                    "skill": skill,
                    "level": "Follow-up",
                    "answer": gap_answer_text,
                    "reference_answer": gap_ref_answer,
                    "relevance_score": gap_score,
                    "missing_keywords": gap_missing,
                    "dominant_emotion": "pending",
                    "audio_emotion": ""
                })

                question_id += 1
                total_questions += 1

        # Stop outer loop if max reached
        if total_questions >= MAX_TOTAL_QUESTIONS:
            break

    # =========================
    # SAVE RESULTS
    # =========================
    output_file = os.path.join(OUTPUT_DIR, "interview_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\nInterview Completed Successfully!")
    print(f"Results saved at: {output_file}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    start_interview()