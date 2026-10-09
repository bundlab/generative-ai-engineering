import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(sentences: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return embeddings

def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b))

if __name__ == "__main__":
    docs = [
        "Dense embeddings capture semantic meaning in vector space.",
        "Cosine similarity measures the angle between directional vectors."
    ]
    vectors = generate_embeddings(docs)
    sim = calculate_cosine_similarity(vectors[0], vectors[1])
    
    print(f"Vector Dimensions: {vectors.shape[1]}")
    print(f"Cosine Similarity Score: {sim:.4f}")
