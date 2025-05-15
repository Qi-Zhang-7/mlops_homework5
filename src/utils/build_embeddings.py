# src/utils/build_embeddings.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
data_path = "./data/6000_all_categories_questions_with_excerpts.csv"  # adjust if needed
output_file = "./data/embeddings.pkl"

def generate_embed(texts, model):
    return model.encode(texts, show_progress_bar=True)

def build_and_save_embeddings():
    # load data
    df = pd.read_csv(data_path)


    excerpts = df["wikipedia_excerpt"].tolist()
    embeddings = generate_embed(excerpts, model)
    df["embedding"] = embeddings.tolist()  # save as list so it can go in .pkl

    df.to_pickle(output_file)
  
if __name__ == "__main__":
    build_and_save_embeddings()