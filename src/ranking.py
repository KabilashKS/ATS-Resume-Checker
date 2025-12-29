import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rank_candidates(
    job_index,
    resumes,
    resume_embeddings,
    job_embeddings,
    top_n=10,
    min_similarity=0.25
):
    # Extract embeddings
    job_vec = job_embeddings[job_index].cpu().numpy().reshape(1, -1)
    resume_vec = resume_embeddings.cpu().numpy()

    # Semantic similarity
    semantic_score = cosine_similarity(job_vec, resume_vec)[0]

    # Filter low semantic matches
    valid_mask = semantic_score >= min_similarity

    # Experience normalization
    exp_score = resumes['experience_years'].astype(float)
    max_exp = exp_score.max()

    if max_exp > 0:
        exp_score = exp_score / max_exp
    else:
        exp_score = np.zeros(len(exp_score))

    # Final weighted score
    final_score = np.zeros(len(semantic_score))
    final_score[valid_mask] = (
        0.85 * semantic_score[valid_mask] +
        0.15 * exp_score[valid_mask]
    )

    # Rank top candidates
    ranked_idx = np.argsort(final_score)[::-1][:top_n]

    result = resumes.iloc[ranked_idx][
        ['resume_id', 'category', 'experience_years']
    ].copy()

    result['match_score'] = final_score[ranked_idx]

    return result
