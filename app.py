import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
from pathlib import Path

from src.preprocess import preprocess_resumes, clean_text, extract_experience
from src.embedding import encode_text
from src.ranking import rank_candidates
from src.resume_parser import parse_resume
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="ATS Resume Checker",
    layout="wide"
)

st.title("📄 ATS Resume Screening & Ranking System")
st.write("Semantic AI-based Resume Shortlisting for HR")

BASE_DIR = Path(__file__).parent


# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    resumes = pd.read_csv(BASE_DIR / "data/resumes.csv")
    jobs = pd.read_csv(BASE_DIR / "data/jobs.csv")
    return resumes, jobs

resumes, jobs = load_data()

# -------------------------------
# Preprocess
# -------------------------------
resumes = preprocess_resumes(resumes)

# -------------------------------
# Embedding Setup
# -------------------------------
EMB_DIR = BASE_DIR / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

resume_emb_path = EMB_DIR / "resume_embeddings.pt"
job_emb_path = EMB_DIR / "job_embeddings.pt"

@st.cache_resource
def load_embeddings():
    if resume_emb_path.exists():
        resume_embeddings = torch.load(resume_emb_path)
    else:
        resume_embeddings = encode_text(resumes["clean_text"].tolist())
        torch.save(resume_embeddings, resume_emb_path)

    if job_emb_path.exists():
        job_embeddings = torch.load(job_emb_path)
    else:
        job_embeddings = encode_text(jobs["job_description"].tolist())
        torch.save(job_embeddings, job_emb_path)

    return resume_embeddings, job_embeddings

resume_embeddings, job_embeddings = load_embeddings()

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("🔍 HR Controls")

job_title_col = "Job Title" if "Job Title" in jobs.columns else "job_title"

job_index = st.sidebar.selectbox(
    "Select Job Role",
    jobs.index,
    format_func=lambda x: jobs.loc[x, job_title_col]
)

top_n = st.sidebar.slider("Number of Candidates", 5, 50, 10)

# -------------------------------
# Rank Candidates
# -------------------------------
if st.sidebar.button("🚀 Rank Candidates"):
    results = rank_candidates(
        job_index,
        resumes,
        resume_embeddings,
        job_embeddings,
        top_n
    )

    st.subheader("🏆 Top Ranked Candidates")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "⬇ Download Shortlist",
        results.to_csv(index=False),
        "shortlisted_candidates.csv",
        "text/csv"
    )
else:
    st.info("👈 Select job role and click Rank Candidates")

# ======================================================
# PUBLIC ATS CHECK
# ======================================================
st.markdown("---")
st.subheader("📤 Upload Resume (Public ATS Check)")

uploaded_resume = st.file_uploader(
    "Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

if uploaded_resume:
    resume_text = parse_resume(uploaded_resume)
    resume_text = clean_text(resume_text)

    experience_years = extract_experience(resume_text)

    resume_vec = encode_text(resume_text).cpu().numpy().reshape(1, -1)
    job_vec = job_embeddings[job_index].cpu().numpy().reshape(1, -1)

    semantic_score = cosine_similarity(job_vec, resume_vec)[0][0]

    max_exp = resumes["experience_years"].max()
    exp_score = experience_years / max_exp if max_exp > 0 else 0

    final_score = 0.85 * semantic_score + 0.15 * exp_score

    col1, col2 = st.columns(2)
    col1.metric("📈 ATS Match Score", f"{final_score*100:.1f}%")
    col2.metric("🧠 Experience (Years)", experience_years)

    st.progress(min(final_score, 1.0))

    if final_score >= 0.75:
        st.success("✅ Strong Match – Highly Recommended")
    elif final_score >= 0.45:
        st.warning("⚠ Moderate Match – Needs Review")
    else:
        st.error("❌ Low Match – Not Suitable")

