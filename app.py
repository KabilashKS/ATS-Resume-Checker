import streamlit as st
import pandas as pd
import torch
import numpy as np
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


# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    resumes = pd.read_csv("data/resumes.csv")
    jobs = pd.read_csv("data/jobs.csv")
    return resumes, jobs


resumes, jobs = load_data()


# -------------------------------
# Preprocess Resumes
# -------------------------------
resumes = preprocess_resumes(resumes)


# -------------------------------
# Load / Create Embeddings (CPU SAFE)
# -------------------------------
@st.cache_resource
def load_embeddings(resumes, jobs):
    emb_dir = Path("embeddings")
    emb_dir.mkdir(exist_ok=True)

    resume_emb_path = emb_dir / "resume_embeddings.pt"
    job_emb_path = emb_dir / "job_embeddings.pt"

    # Resume embeddings
    if resume_emb_path.exists():
        resume_embeddings = torch.load(
            resume_emb_path, map_location=torch.device("cpu")
        )
    else:
        resume_embeddings = encode_text(resumes["clean_text"].tolist())
        torch.save(resume_embeddings, resume_emb_path)

    # Job embeddings
    job_col = "job_description" if "job_description" in jobs.columns else "description"

    if job_emb_path.exists():
        job_embeddings = torch.load(
            job_emb_path, map_location=torch.device("cpu")
        )
    else:
        job_embeddings = encode_text(jobs[job_col].tolist())
        torch.save(job_embeddings, job_emb_path)

    return resume_embeddings, job_embeddings


resume_embeddings, job_embeddings = load_embeddings(resumes, jobs)


# -------------------------------
# Sidebar (HR Controls)
# -------------------------------
st.sidebar.header("🔍 HR Controls")

job_title_col = "Job Title" if "Job Title" in jobs.columns else "job_title"

job_index = st.sidebar.selectbox(
    "Select Job Role",
    jobs.index,
    format_func=lambda x: jobs.loc[x, job_title_col]
)

top_n = st.sidebar.slider(
    "Number of Candidates",
    min_value=5,
    max_value=50,
    value=10
)


# -------------------------------
# Rank Existing Candidates
# -------------------------------
if st.sidebar.button("🚀 Rank Candidates"):

    results = rank_candidates(
        job_index=job_index,
        resumes=resumes,
        resume_embeddings=resume_embeddings,
        job_embeddings=job_embeddings,
        top_n=top_n
    )

    st.subheader("🏆 Top Ranked Candidates")

    st.dataframe(
        results.style.format({"match_score": "{:.2f}"}),
        use_container_width=True
    )

    csv = results.to_csv(index=False)
    st.download_button(
        "⬇ Download Shortlist",
        csv,
        file_name="shortlisted_candidates.csv",
        mime="text/csv"
    )
else:
    st.info("👈 Select job role and click **Rank Candidates**")


# ======================================================
# PUBLIC RESUME UPLOAD (CANDIDATE ATS CHECK)
# ======================================================
st.markdown("---")
st.subheader("📤 Upload Resume (Public ATS Check)")

uploaded_resume = st.file_uploader(
    "Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

if uploaded_resume is not None:
    st.subheader("📊 ATS Match Result")

    resume_text = parse_resume(uploaded_resume)
    resume_text = clean_text(resume_text)

    experience_years = extract_experience(resume_text)

    resume_embedding = encode_text(resume_text).cpu().numpy().reshape(1, -1)
    job_vec = job_embeddings[job_index].cpu().numpy().reshape(1, -1)

    semantic_score = cosine_similarity(job_vec, resume_embedding)[0][0]

    max_exp = resumes["experience_years"].max()
    exp_score = experience_years / max_exp if max_exp > 0 else 0

    final_score = 0.85 * semantic_score + 0.15 * exp_score

    col1, col2 = st.columns(2)
    col1.metric("📈 Match Score", f"{final_score:.2f}")
    col2.metric("🧠 Experience (Years)", experience_years)

    if final_score >= 0.75:
        st.success("✅ Strong Match – Highly Recommended")
    elif final_score >= 0.45:
        st.warning("⚠ Moderate Match – Needs Review")
    else:
        st.error("❌ Low Match – Not Suitable")

