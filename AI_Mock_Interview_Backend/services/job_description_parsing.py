import os
import json
import pandas as pd
import re

# =====================================================
# 1️⃣ SAMPLE JOB DESCRIPTION (You can replace this)
# =====================================================

job_description = """
We are hiring a Full Stack Developer with strong experience in Python, Django,
React, and Node.js. The candidate should have experience working with SQL,
GraphQL, and REST APIs. Knowledge of Docker, MongoDB, and cloud deployment
is a plus. Familiarity with WebSocket and Redux is preferred.
"""

# =====================================================
# 2️⃣ DEFINE PROJECT ROOT PATH
# =====================================================

# Get current file directory (service folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up to reach project root
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Define dataset and output paths
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "technical_skills.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "extracted_skills.json")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 3️⃣ LOAD TECHNICAL SKILLS CSV
# =====================================================

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print("❌ technical_skills.csv not found in dataset folder.")
    exit()

required_columns = {"Skill ID", "Skill Name", "Category"}

if not required_columns.issubset(df.columns):
    print("❌ CSV must contain: Skill ID, Skill Name, Category")
    exit()

# =====================================================
# 4️⃣ PREPROCESS JOB DESCRIPTION
# =====================================================

jd_text = job_description.lower()

# =====================================================
# 5️⃣ SKILL EXTRACTION
# =====================================================

matched_skills = []

for _, row in df.iterrows():
    skill_id = row["Skill ID"]
    skill_name = str(row["Skill Name"])
    category = row["Category"]

    # Escape special characters (C++, C#, .NET, etc.)
    skill_pattern = re.escape(skill_name.lower())

    # Regex pattern for full match
    pattern = r"(?<!\w)" + skill_pattern + r"(?!\w)"

    if re.search(pattern, jd_text):
        matched_skills.append({
            "Skill ID": int(skill_id),
            "Skill Name": skill_name,
            "Category": category
        })

# Remove duplicates
unique_skills = {skill["Skill Name"]: skill for skill in matched_skills}
matched_skills = list(unique_skills.values())

# =====================================================
# 6️⃣ OUTPUT JSON STRUCTURE
# =====================================================

output_data = {
    "job_description": job_description,
    "total_skills_matched": len(matched_skills),
    "extracted_skills": matched_skills
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)

# =====================================================
# 7️⃣ PRINT OUTPUT
# =====================================================

print("\n✅ Extracted Skills:\n")

for skill in matched_skills:
    print(f"{skill['Skill ID']} - {skill['Skill Name']} ({skill['Category']})")

print(f"\n📁 Output saved to: {OUTPUT_FILE}")