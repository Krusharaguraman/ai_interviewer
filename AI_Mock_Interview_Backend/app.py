import time

from services import job_description_parsing
from services import core_interview
from services import video_emotion_analyzer
from services import audio_emotion_analyzer


def main():
    print("\n==============================")
    print(" AI MOCK INTERVIEW SYSTEM ")
    print("==============================\n")

    # Step 1
    job_description_parsing.run()
    time.sleep(1)

    # Step 2
    core_interview.start_interview()
    time.sleep(1)

    # Step 3
    video_emotion_analyzer.run()
    time.sleep(1)

    # Step 4
    audio_emotion_analyzer.run()

    print("\n🎉 FULL MOCK INTERVIEW PROCESS COMPLETED!\n")


if __name__ == "__main__":
    main()