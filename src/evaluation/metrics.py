"""
Evaluation metrics for similarity and clustering.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evaluate_similarity(predictions: list[float], gold_scores: np.ndarray) -> dict[str, float]:
    """
    Evaluate similarity predictions against gold scores.
    Returns Pearson and Spearman correlations.
    """
    pearson_corr, _ = pearsonr(predictions, gold_scores)
    spearman_corr, _ = spearmanr(predictions, gold_scores)
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }

def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Evaluate clustering quality using silhouette score.
    """
    score = silhouette_score(embeddings, labels)
    return score

def compute_tfidf_similarity(text1: str, text2: str, vectorizer: TfidfVectorizer) -> float:
    """
    Compute TF-IDF cosine similarity between two texts.
    """
    tfidf1 = vectorizer.transform([text1]).toarray()[0]
    tfidf2 = vectorizer.transform([text2]).toarray()[0]
    return cosine_sim(tfidf1, tfidf2)