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

    # Step 3: Format top results
    return [
        AnswerItem(
            text=texts[i],
            score=float(distances[0][j])
        )
        for j, i in enumerate(indices[0])
    ]