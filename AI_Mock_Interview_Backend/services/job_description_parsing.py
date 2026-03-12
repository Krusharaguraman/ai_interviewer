import os
import json
import pandas as pd
import re

# =====================================================
# PATH SETUP
# =====================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "technical_skills.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "extracted_skills.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# SKILL EXTRACTION FUNCTION
# =====================================================

def extract_skills_from_jd(job_description):

    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print("❌ technical_skills.csv not found.")
        return []

    required_columns = {"Skill ID", "Skill Name", "Category"}

    if not required_columns.issubset(df.columns):
        print("❌ CSV must contain: Skill ID, Skill Name, Category")
        return []

    jd_text = job_description.lower()

    matched_skills = []

    for _, row in df.iterrows():

        skill_id = row["Skill ID"]
        skill_name = str(row["Skill Name"])
        category = row["Category"]

        skill_pattern = re.escape(skill_name.lower())
        pattern = r"(?<!\w)" + skill_pattern + r"(?!\w)"

        if re.search(pattern, jd_text):

            matched_skills.append({
                "Skill ID": int(skill_id),
                "Skill Name": skill_name,
                "Category": category
            })

    # remove duplicates
    unique_skills = {skill["Skill Name"]: skill for skill in matched_skills}
    matched_skills = list(unique_skills.values())

    output_data = {
        "job_description": job_description,
        "total_skills_matched": len(matched_skills),
        "extracted_skills": matched_skills
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print("\n✅ Extracted Skills:\n")

    for skill in matched_skills:
        print(f"{skill['Skill ID']} - {skill['Skill Name']} ({skill['Category']})")

    print(f"\n📁 Output saved to: {OUTPUT_FILE}")

    # return skill names list for interview
    return [skill["Skill Name"] for skill in matched_skills]


# =====================================================
# TEST MODE (Run file directly)
# =====================================================

if __name__ == "__main__":

    jd = input("\nEnter Job Description:\n")

    skills = extract_skills_from_jd(jd)

    print("\nSkills detected:", skills)