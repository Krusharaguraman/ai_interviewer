import os
import subprocess
import sys

# verify expected dependencies - this makes main.py more forgiving when run
# from a fresh environment.  It does not install anything automatically, but
# prints a helpful message if import fails.
def _check_import(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except ImportError:
        print(f"⚠️  Missing package '{pkg_name}'.  Please run `pip install -r requirements.txt`.")

# run checks for the packages we know the services use
for pkg in [
    "pandas",
    "numpy",
    "sounddevice",
    "soundfile",
    "opencv-python",
    "sentence_transformers",
    "faster_whisper",
    "librosa",
    "sklearn",
]:
    _check_import(pkg)

# ===============================
# IMPORT SERVICES
# ===============================
from services.job_description_parsing import extract_skills_from_jd
from services.core_interview import start_interview
from services.audio_emotion_analyzer import analyze_audio_emotions
from services.video_emotion_analyzer import analyze_video_emotions
from services.final_feedback_report import generate_final_report
# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_EMOTION_SCRIPT = os.path.join(BASE_DIR, "services", "video_emotion_analyzer.py")
FINAL_FEEDBACK_SCRIPT = os.path.join(BASE_DIR, "services", "final_feedback_report.py")


# ===============================
# PERMISSION PROMPT
# ===============================
def ask_permissions():

    print("\n🎤 Camera and Microphone access required")

    cam = input("Allow Camera access? (yes/no): ").lower()
    mic = input("Allow Microphone access? (yes/no): ").lower()

    if cam != "yes" or mic != "yes":
        print("\n❌ Camera/Microphone permission denied.")
        print("Interview cannot continue.")
        exit()

    print("\n✅ Permissions granted\n")


# ===============================
# MAIN PIPELINE
# ===============================
def main():

    print("\n==============================")
    print("AI MOCK INTERVIEW SYSTEM")
    print("==============================\n")

    # ===============================
    # STEP 1 : GET JOB DESCRIPTION
    # ===============================
    jd = input("Paste Job Description:\n\n")

    skills = extract_skills_from_jd(jd)

    if not skills:
        print("❌ No skills detected. Exiting.")
        return

    # ===============================
    # STEP 2 : ASK CAMERA + MIC PERMISSION
    # ===============================
    ask_permissions()

    # ===============================
    # STEP 3 : START INTERVIEW
    # ===============================
    print("\n🎙 Starting Interview...\n")

    start_interview()

    # ===============================
    # STEP 4 : AUDIO EMOTION ANALYSIS
    # ===============================
    print("\n🎧 Analyzing Audio Emotions...\n")

    analyze_audio_emotions()

    # ===============================
    # STEP 5 : VIDEO EMOTION ANALYSIS
    # ===============================
    print("\n📷 Analyzing Video Emotions...\n")

    # ensure the same Python interpreter is used when invoking secondary
    # scripts (especially important when a venv is active)
    subprocess.run([sys.executable, VIDEO_EMOTION_SCRIPT])

    # ===============================
    # STEP 6 : FINAL FEEDBACK
    # ===============================
    print("\n📊 Generating Final Feedback...\n")

    subprocess.run([sys.executable, FINAL_FEEDBACK_SCRIPT])

    print("\n✅ Interview Process Completed Successfully!")


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    main()