import pandas as pd
from src.preprocess import clean_text
from src.preprocess import preprocess_resumes
import torch
from src.embedding import generate_embeddings
from src.ranking import rank_candidates

resumes = pd.read_csv('data/resumes.csv')
jobs = pd.read_csv('data/jobs.csv')

resumes.columns = resumes.columns.str.strip().str.lower()
jobs.columns = jobs.columns.str.strip().str.lower()

resumes = preprocess_resumes(resumes)

jobs['job_final'] = jobs['Job Title'].astype(str) + " " + jobs['job description'].astype(str)

resume_embeddings = generate_embeddings(
    resumes['resume_final'].tolist(),
    "embeddings/resume_embeddings.pt"
)

job_embeddings = generate_embeddings(
    jobs['job_final'].tolist(),
    "embeddings/job_embeddings.pt"
)

resume_embeddings = torch.load("embeddings/resume_embeddings.pt")
job_embeddings = torch.load("embeddings/job_embeddings.pt")

top_candidates = rank_candidates(
    job_index = 0,
    resumes = resumes,
    resume_embeddings = resume_embeddings,
    job_embeddings = job_embeddings,
    top_n = 5
)

print(top_candidates)
















