from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts, save_path):
    embeddings = model.encode(
        texts,
        convert_to_tensor = True,
        show_progress_bar = True
    )
    torch.save(embeddings, save_path)
    return embeddings


def encode_text(text):
    return model.encode(text, convert_to_tensor=True)