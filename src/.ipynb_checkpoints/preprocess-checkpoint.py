import re
import pandas as pd
from datetime import datetime

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'[\s]', ' ', text)
    return text.strip()

def extract_experience(text):
    if pd.isna(text) or text.strip() == "":
        return 0.0

    text = text.lower()
    current_year = datetime.now().year

    # 1️⃣ Explicit: "3 years", "5+ yrs"
    explicit = re.findall(r'(\d+)\+?\s*(years|yrs|year)', text)
    if explicit:
        return float(explicit[0][0])

    # 2️⃣ Date ranges
    ranges = re.findall(
        r'(19\d{2}|20\d{2})\s*(?:-|–|to)\s*(19\d{2}|20\d{2}|present)',
        text
    )
    durations = []
    for start, end in ranges:
        start = int(start)
        end = current_year if end == "present" else int(end)
        if end >= start:
            durations.append(end - start)

    if durations:
        return float(max(durations))

    # 3️⃣ Heuristic keywords (VERY IMPORTANT)
    senior_keywords = {
        "intern": 0,
        "junior": 1,
        "fresher": 0,
        "associate": 2,
        "developer": 2,
        "engineer": 3,
        "senior": 5,
        "lead": 6,
        "manager": 7,
        "architect": 8
    }

    for keyword, years in senior_keywords.items():
        if keyword in text:
            return float(years)

    return 1.0  # default minimum experience

def preprocess_resumes(resumes):
    resumes['resume_text'] = resumes['resume_text'].fillna("").apply(clean_text)
    resumes['skills_list'] = resumes['skills_list'].fillna("").apply(clean_text)

    resumes['resume_final'] = (
        resumes['resume_text'] + " " + resumes['skills_list']
    )

    resumes['experience_years'] = resumes['resume_final'].apply(extract_experience)

    return resumes





        





        
