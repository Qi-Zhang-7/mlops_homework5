# from fastapi import FastAPI
# from src.api import query
# from fastapi.responses import RedirectResponse



# app = FastAPI(
#     title="ML API",
#     description="API for ML Model Inference",
#     version="1.0.0",
# )

# @app.get("/")
# async def redirect_to_docs():
#     return RedirectResponse(url="/docs")

# app.include_router(query.router)

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api import query
import pandas as pd
import numpy as np

# Create FastAPI app
app = FastAPI(
    title="ML API",
    description="API for ML Model Inference",
    version="1.0.0",
)

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# === Load resources ===

from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model (can be reused across requests)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
app.state.embedding_model = model

# Load prebuilt pickle file
df = pd.read_pickle("data/embeddings.pkl")

# Store raw texts and embeddings
app.state.texts = df["wikipedia_excerpt"].tolist()
app.state.embeddings = np.vstack(df["embedding"].values).astype("float32")

# Optional: if you want to load FAISS instead of cosine, you could load the index here
# import faiss
# app.state.index = faiss.read_index("data/wiki_index_hnsw.faiss")

# Register routes
app.include_router(query.router)