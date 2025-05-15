import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.models.query import AnswerItem


# Load embedding model (once)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load precomputed data
df = pd.read_pickle("data/embeddings.pkl")


# Extract texts and embeddings
texts = df["wikipedia_excerpt"].tolist()
embeddings = np.vstack(df["embedding"].values).astype('float32')  # FAISS requires float32

# Build FAISS index (HNSW)
dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efConstruction = 200
index.add(embeddings)  # Add all vectors

def get_similar_responses(question: str, top_k: int = 3) -> list[AnswerItem]:
    # Step 1: Convert input question to embedding

    query_embedding = model.encode([question], convert_to_numpy=True).astype('float32')

    # Step 2: Search HNSW index
    distances, indices = index.search(query_embedding, top_k)
    
    # Optional: define a similarity score threshold (adjust as needed)
    min_score_threshold = 0.2  # FAISS returns L2 or inner product depending on index type

    results = []
    for j, i in enumerate(indices[0]):
        excerpt = texts[i].strip()
        score = float(distances[0][j])
        if excerpt and score > min_score_threshold:
            results.append(AnswerItem(text=excerpt, score=score))

    # Step 3: Fallback if no good results
    if not results:
        results = [AnswerItem(text="No relevant information found.", score=None)]

    return results
