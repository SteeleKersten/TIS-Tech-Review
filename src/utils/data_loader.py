"""
Data loading utilities for the project.
"""

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def load_stsb_dataset(split: str = 'train') -> pd.DataFrame:
    """
    Load STSB dataset (Semantic Textual Similarity Benchmark).
    split: 'train', 'dev', or 'test'
    """
    dataset = load_dataset("stsb_multi_mt", "en", split=split)
    df = pd.DataFrame({
        'sentence1': dataset['sentence1'],
        'sentence2': dataset['sentence2'],
        'similarity_score': dataset['similarity_score']
    })
    return df

def load_clustering_dataset() -> pd.DataFrame:
    """
    Load AG News dataset for clustering (subset for short documents).
    Returns DataFrame with text and labels.
    """
    dataset = load_dataset("ag_news", split="train")
    df = pd.DataFrame({
        'text': dataset['text'],
        'label': dataset['label']
    })
    # Take a subset for faster processing
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    return df

def get_sbert_embeddings(texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Generate SBERT embeddings for a list of texts.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def get_tfidf_embeddings(texts: list[str], max_features: int = 5000) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Generate TF-IDF embeddings for a list of texts.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray(), vectorizer